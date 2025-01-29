import os
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from transformers.trainer import get_scheduler



from openrlhf.datasets.offline_ppo_dataset import OfflinePPODataset
from openrlhf.trainer.ppo_utils import OfflineExperienceMaker, FixedKLController
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils import get_strategy, get_tokenizer, blending_datasets
from openrlhf.trainer import PPOTrainer

@dataclass
class OfflinePPOArgs:
    # Existing args from original PPO
    pretrain: str = "meta-llama/Meta-Llama-3-8B"
    reward_pretrain: str = "OpenRLHF/Llama-3-8b-rm-mixture"
    save_path: str = "./checkpoint/llama-3-8b-rlhf"
    max_epochs: int = 1
    micro_train_batch_size: int = 2
    # New offline-specific args
    max_prompt_length: int = 1024
    max_completion_length: int = 1024
    offline_data_path: str = "./data/offline_rlhf.json"

def train_offline(args):
    # Initialize strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()
    strategy.args = args

    # First load the base model for tokenizer setup
    base_model = AutoModelForCausalLM.from_pretrained(
        args.pretrain,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    # Load tokenizer with base model
    tokenizer = get_tokenizer(
        args.pretrain,
        model=base_model,  # Use base Hugging Face model
        padding_side="left",
        strategy=strategy,
        use_fast=not args.disable_fast_tokenizer,
    )
    
    # Now load the actual actor model
    actor = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )
    
    # Clean up base model
    del base_model

    # Load models
    critic = get_llm_for_sequence_regression(
        args.reward_pretrain,
        "reward",
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_train_config(is_actor=False),
    )

    # Load offline dataset
    offline_data = blending_datasets(
        args.offline_data_path,
        args.prompt_data_probs,
        strategy=strategy
    )
    dataset = OfflinePPODataset(
        data=offline_data,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_train_batch_size,
        collate_fn=dataset.collate_fn,  # You'll need to implement this
        shuffle=True
    )

    # Initialize KL controller
    kl_controller = FixedKLController(args.init_kl_coef)

    # Initialize experience maker with required components
    exp_maker = OfflineExperienceMaker(
        actor=actor,
        critic=critic,
        reward_model=None,
        initial_model=actor,
        prompt_max_len=args.prompt_max_len,
        kl_controller=kl_controller,
        tokenizer=tokenizer
    )

    # Initialize PPO trainer
    trainer = PPOTrainer(
        strategy=strategy,
        actor=actor,
        critic=critic,
        reward_model=None,  # Not needed since rewards are precomputed
        experience_maker=exp_maker,
        tokenizer=tokenizer,
        max_epochs=args.max_epochs,
        # Add other PPOTrainer arguments from original implementation
    )

    # After actor initialization
    initial_model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=False),
    )

    if args.enable_ema:
        ema_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=True),
        )
    else:
        ema_model = None

    # After model creation
    actor_optim = strategy.create_optimizer(
        actor, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
    )

    if args.critic_pretrain:
        critic_optim = strategy.create_optimizer(
            critic, lr=args.critic_learning_rate, betas=args.adam_betas, weight_decay=args.l2
        )
    else:
        critic_optim = None

    # After optimizer creation
    total_steps = args.max_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        "cosine",
        actor_optim,
        num_warmup_steps=int(total_steps * args.lr_warmup_ratio),
        num_training_steps=total_steps,
    )

    # Start training
    trainer.fit(
        args,
        prompts_dataloader=dataloader,  # Our offline dataset acts as prompts
        pretrain_dataloader=None  # Disable pretraining if not needed
    )

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    # Add ALL arguments from original train_ppo.py manually
    # (Copy these from the original train_ppo.py's argument definitions)
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=3e-6)
    parser.add_argument("--init_kl_coef", type=float, default=0.01)
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--input_template", type=str, default=None)
    
    # Add new offline-specific arguments
    parser.add_argument("--offline_data_path", type=str, required=True)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    
    # Add these missing arguments from original train_ppo.py
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--ptx_coef", type=float, default=0.05)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--value_clip", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--lambd", type=float, default=1.0)
    parser.add_argument("--use_kl_estimator_k3", action="store_true", default=False)
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10))
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--enable_ema", action="store_true", default=False)
    parser.add_argument("--zpg", type=int, default=1)
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--prompt_data_probs", type=str, default="1.0")
    parser.add_argument("--pretrain_data_probs", type=str, default="1.0")
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"))
    parser.add_argument("--use_tensorboard", type=str, default=None)
    
    # Add max_samples argument
    parser.add_argument("--max_samples", type=int, default=100000)
    
    # Add local_rank argument
    parser.add_argument("--local_rank", type=int, default=-1)
    
    
    args, _ = parser.parse_known_args()
    train_offline(args) 