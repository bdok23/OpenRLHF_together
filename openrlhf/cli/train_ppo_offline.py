import os
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from transformers.trainer import get_scheduler
from torch.optim import Adam
from transformers import get_cosine_schedule_with_warmup
from argparse import ArgumentParser
import math
import itertools

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.datasets.offline_ppo_dataset import OfflinePPODataset
from openrlhf.trainer.ppo_utils import OfflineExperienceMaker, FixedKLController
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils import get_strategy, get_tokenizer, blending_datasets
from openrlhf.trainer import PPOTrainer

# idk what the below dataclass is used for, i think I already have the args in the parser
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

    actor = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )
    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())
        
    if args.critic_pretrain:
        critic = get_llm_for_sequence_regression(
            args.critic_pretrain,
            "critic",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=args.value_head_prefix,
            init_value_head=strategy.args.pretrain == strategy.args.critic_pretrain,
        )
    else:
        critic = None
        
    if not args.remote_rm_url:
        reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=args.value_head_prefix,
        )
    else:
        reward_model = None
    
    strategy.print("reward normalization status: {}".format(args.normalize_reward))
    if reward_model:
        strategy.print("mean: {}, std {}".format(reward_model.mean, reward_model.std))

    strategy.print(actor)
    strategy.print(critic)
    
    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, actor.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # load weights for reference actor
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
        
    # gradient_checkpointing
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
        if critic is not None:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

    # configure optimizer
    actor_optim = strategy.create_optimizer(
        actor, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
    )
    if args.critic_pretrain:
        critic_optim = strategy.create_optimizer(
            critic, lr=args.critic_learning_rate, betas=args.adam_betas, weight_decay=args.l2
        )
    else:
        critic_optim = None



    # Load offline dataset
    offline_data = blending_datasets(
        args.offline_data_path,
        args.prompt_data_probs,
        strategy,
        # args.seed,
        max_count=args.max_samples,
        return_eval=False,
        # train_split=args.prompt_split,
    )
    offline_data = offline_data.select(range(min(args.max_samples, len(offline_data))))

    dataset = OfflinePPODataset(
        data=offline_data,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length
    )
    
    # if args.pretrain_data:
    #     pretrain_data = blending_datasets(
    #         args.pretrain_data,
    #         args.pretrain_data_probs,
    #         strategy,
    #         args.seed,
    #         return_eval=False,
    #         train_split=args.pretrain_split,
    #     )
    #     pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
    #     pretrain_dataset = SFTDataset(
    #         pretrain_data.select(
    #             range(min(len(pretrain_data), args.max_epochs * len(prompts_dataset) * args.n_samples_per_prompt))
    #         ),
    #         tokenizer,
    #         pretrain_max_len,
    #         strategy,
    #         pretrain_mode=True,
    #     )
    
    # Create dataloader
    dataloader = strategy.setup_dataloader(
        dataset,
        batch_size=args.micro_train_batch_size,  # Directly use training batch size
        shuffle=True,
        collate_fn=dataset.collate_fn  # Use dataset's collate function
    )
    
    # if args.pretrain_data:
    #     pretrain_dataloader = itertools.cycle(
    #         iter(
    #             strategy.setup_dataloader(
    #                 pretrain_dataset,
    #                 args.micro_train_batch_size,
    #                 True,
    #                 True,
    #                 pretrain_dataset.collate_fn,
    #             )
    #         )
    #     )
    # else:
    #     pretrain_dataloader = None
    
    
    # configure scheduler
    num_update_steps_per_episodes = (
        len(dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
    )
    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)

    actor_scheduler = get_scheduler(
        "cosine_with_min_lr",
        actor_optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
    )

    if args.critic_pretrain:
        critic_scheduler = get_scheduler(
            "cosine_with_min_lr",
            critic_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.critic_learning_rate * 0.1},
        )
    else:
        critic_scheduler = None
        
    # prepare models/optimizers...
    (
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        initial_model,
    ) = strategy.prepare(
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        initial_model,
        is_rlhf=True,
    )
    if ema_model:
        ema_model._offload = True
        ema_model = strategy.prepare(ema_model, is_rlhf=True)


    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_actor")):
        _, states = strategy.load_ckpt(actor.model, os.path.join(args.ckpt_path, "_actor"))
        if args.critic_pretrain:
            strategy.load_ckpt(critic, os.path.join(args.ckpt_path, "_critic"))
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)
    
    
    # Initialize KL controller
    kl_controller = FixedKLController(args.init_kl_coef)

    # Initialize experience maker with required components
    exp_maker = OfflineExperienceMaker(
        actor=actor,
        critic=critic,
        reward_model=reward_model,
        initial_model=initial_model,
        tokenizer=tokenizer,
        prompt_max_len=args.prompt_max_len,
        kl_controller=kl_controller,
        strategy=strategy,
        remote_rm_url=args.remote_rm_url,
        reward_fn=None,
        offline_dataset=dataset,
        # device=strategy.device, # deepspeed strategy has no device attribute
        # advantage_estimator=strategy.args.advantage_estimator,
    )
    
    # Initialize PPO trainer
    trainer = PPOTrainer(
        strategy=strategy,
        actor=actor,
        critic=critic,
        reward_model=reward_model,
        initial_model=initial_model,
        ema_model=ema_model,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        max_epochs=args.max_epochs,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        prompt_max_len=args.prompt_max_len,
        value_clip=args.value_clip,
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lambd=args.lambd,
        init_kl_coef=args.init_kl_coef,
        kl_target=args.kl_target,
        ema_beta=0.992,
        ptx_coef=args.ptx_coef,
        max_norm=args.max_norm,
        # for GPT generation
        do_sample=True,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # remote reward model
        remote_rm_url=args.remote_rm_url,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
        
        experience_maker=exp_maker,
    )

    # Start training
    trainer.fit(
        args,
        prompts_dataloader=dataloader,  # Our offline dataset acts as prompts
        pretrain_dataloader=None,  # Disable pretraining if not needed
        consumed_samples=consumed_samples,
        num_update_steps_per_episodes=num_update_steps_per_episodes
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)    
    
    # PPO 
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=100000)  # Offline specific
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=1.0, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6) # is used for offline
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6) # is used for offline
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO") # is used for offline
    parser.add_argument(
        "--use_kl_estimator_k3",
        action="store_true",
        default=False,
        help=(
            "Use the k3 estimator in http://joschu.net/blog/kl-approx.html"
            "to ensure the KL divergence calculated is non-negative"
        ),
    )
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")

    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False) # is used for offline

    # Reinforce
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo",
    )
    
    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0) # is used for offline
    parser.add_argument("--lora_alpha", type=int, default=16) # is used for offline
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear") # is used for offline
    parser.add_argument("--lora_dropout", type=float, default=0) # is used for offline

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path") # is used for offline
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path") # is used for offline
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path") 
    parser.add_argument("--value_head_prefix", type=str, default="score")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path") 
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )   # is used for offline
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key") 
    parser.add_argument("--input_template", type=str, default=None) 
    parser.add_argument("--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template") 

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )
    
    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")


    # New Added Arguemnts for offline (some may or may not be used)
    parser.add_argument("--actor_weight_decay", type=float, default=0.0)
    parser.add_argument("--critic_weight_decay", type=float, default=0.0)
    parser.add_argument("--actor_rho", type=float, default=0.0)
    parser.add_argument("--critic_rho", type=float, default=0.0)
    parser.add_argument("--disable_gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--adap_kl_ctrl", action="store_true", default=False)
    parser.add_argument("--max_prompt_length", type=int, default=1024) # is used for offline
    parser.add_argument("--max_completion_length", type=int, default=1024) # is used for offline
    parser.add_argument("--disable_actor_critic_ckpt", action="store_true", default=False)
    parser.add_argument("--enable_galore", action="store_true", default=False)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--scale_reward", action="store_true", default=False)
    parser.add_argument("--whiten_reward", action="store_true", default=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--disable_ds", action="store_true", default=False)
    parser.add_argument("--offline_data_path", type=str, required=True)  # is used for offline



    args = parser.parse_args()
    train_offline(args) 