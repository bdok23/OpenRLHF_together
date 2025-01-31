import torch
from copy import deepcopy
from .experience_maker import Experience, Samples
from .experience_maker import NaiveExperienceMaker
from typing import List
from .kl_controller import FixedKLController, AdaptiveKLController
from .experience_maker import compute_approx_kl, compute_reward

class OfflineExperienceMaker(NaiveExperienceMaker):
    """
    An 'offline' version of the Experience Maker that
    reads completions and rewards from the dataset, instead of
    generating them on the fly.
    """
    def __init__(
        self,
        actor,
        critic,
        reward_model,
        initial_model,
        prompt_max_len: int,
        kl_controller: FixedKLController,
        tokenizer=None,
        *args,
        **kwargs
    ):
        # Pop offline_dataset from kwargs so it won't be passed to super().__init__()
        self.offline_dataset = kwargs.pop("offline_dataset", None)

        # any local device settings
        self.device = kwargs.pop("device", "cuda")

        super().__init__(
            actor=actor,
            critic=critic,
            reward_model=reward_model,
            initial_model=initial_model,
            prompt_max_len=prompt_max_len,
            kl_controller=kl_controller,
            tokenizer=tokenizer,
            *args,
            **kwargs
        )
        
        self.kl_ctl = kl_controller
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    @torch.no_grad()
    def make_experience_list(self, all_prompts=None, **generate_kwargs):
        """
        Instead of calling model.generate, simply read from self.offline_dataset
        and form the Experience objects with completions + reward already provided.
        
        The 'all_prompts' argument might be unused here, but you may keep it
        for consistency with the parent class method signature.
        """
        experiences = []
        
        for i, (prompt, completion, reward) in enumerate(self.offline_dataset):
            # Build an Experience with your already-known prompt, completion, reward
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            completion_ids = self.tokenizer(completion, return_tensors="pt").input_ids.to(self.device)

            # Concatenate tokens so it looks like [prompt_ids, completion_ids]
            sequences = torch.cat([prompt_ids, completion_ids], dim=1)

            # Build attention_mask
            attention_mask = torch.ones_like(sequences).to(self.device)
            # Build action_mask (1 for completion tokens, 0 for prompt tokens)
            action_mask = torch.zeros_like(sequences).to(self.device)
            action_mask[:, -completion_ids.size(1):] = 1

            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
            )

            # Create Experience object with placeholders for log_probs/values
            exp = Experience(
                sequences=sequences,
                action_log_probs=torch.zeros_like(sequences),  # placeholder
                values=torch.zeros(1, sequences.size(1)),      # placeholder
                returns=None,
                advantages=None,
                attention_mask=attention_mask,
                action_mask=action_mask,
                info={
                    "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                    "response_length": samples.response_length,
                    "total_length": samples.total_length,
                    "num_actions": action_mask.size(1),
                },
                kl=None,  # Will be calculated later
            )

            exp.to_device("cpu")
            experiences.append(exp)

        # Now process experiences using parent's pipeline, which sets up
        # references to advantage calculation etc.
        experiences, rewards = self.process_experiences(experiences)

        # Next, properly calculate advantages/returns 
        for experience, rew_val in zip(experiences, rewards):
            experience = experience.to_device(self.device)
            
            with torch.no_grad():
                # Current policy log probs
                action_log_probs = self.actor(
                    experience.sequences, 
                    experience.info["num_actions"],
                    experience.attention_mask
                )
                
                # Base (initial) policy log probs
                base_action_log_probs = self.initial_model(
                    experience.sequences,
                    experience.info["num_actions"], 
                    experience.attention_mask
                )
                
                # KL divergence
                experience.kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    action_mask=experience.action_mask,
                    use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3
                )

            # final reward = dataset reward - KL * kl_coef (with optional clip)
            reward_tensor = compute_reward(
                rew_val,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=experience.info["num_actions"],
                reward_clip_range=self.strategy.args.reward_clip_range,
            )

            # advantage_estimator logic
            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward_tensor,
                    experience.action_mask,
                    generate_kwargs.get("gamma", 0.99),
                    generate_kwargs.get("lambd", 0.95),
                )
            elif self.advantage_estimator in ["reinforce", "rloo"]:
                experience.returns = self.get_cumulative_returns(
                    reward_tensor,
                    experience.action_mask,
                    generate_kwargs.get("gamma", 0.99),
                )
                experience.advantages = experience.returns

            experience.info["return"] = experience.returns.sum().item()
            experience.to_device("cpu")

        return experiences

    def make_experience_batch(self, batch: dict) -> List[Experience]:
        """
        Example of processing a batch instead of individual (prompt, completion, reward).
        """
        experiences = []
        for prompt_ids, completion_ids, reward in zip(
            batch["prompt_ids"], 
            batch["completion_ids"],
            batch["reward"]
        ):
            input_ids = torch.cat([prompt_ids, completion_ids], dim=-1)
            
            attention_mask = torch.ones_like(input_ids)
            action_mask = torch.zeros_like(input_ids)
            action_mask[len(prompt_ids):] = 1
            
            samples = Samples(
                sequences=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                action_mask=action_mask.unsqueeze(0),
                num_actions=len(completion_ids),
                packed_seq_lens=None,
                response_length=len(completion_ids),
                total_length=len(input_ids),
            )
            
            # For now, placeholders:
            exp = Experience(
                sequences=input_ids.unsqueeze(0),
                action_log_probs=torch.zeros(1, input_ids.size(-1)),
                values=torch.zeros(1, input_ids.size(-1)),
                returns=None,
                advantages=None,
                attention_mask=attention_mask.unsqueeze(0),
                action_mask=action_mask.unsqueeze(0),
                info={
                    "reward": torch.tensor([reward], dtype=torch.float32),
                    "num_actions": len(completion_ids)
                },
                kl=None
            )
            
            experiences.append(exp)
        
        return experiences
