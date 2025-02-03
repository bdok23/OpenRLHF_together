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
        
        ###
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
        if self.offline_dataset is None:
            # if no dataset was provided in kwargs, do nothing
            return []

        experiences = []
        # build raw Experience objects from offline_dataset
        for i, (prompt, completion, reward) in enumerate(self.offline_dataset):
            # ---------------
            # 1) Tokenize
            # ---------------
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            completion_ids = self.tokenizer(completion, return_tensors="pt").input_ids.to(self.device)

            # sequences = [prompt, completion], shape (1, S)
            sequences = torch.cat([prompt_ids, completion_ids], dim=1)

            # ---------------
            # 2) Masks
            # ---------------
            attention_mask = torch.ones_like(sequences).to(self.device)  # (1, S)
            action_mask = torch.zeros_like(sequences).to(self.device)    # (1, S)
            # Mark the completion portion as the "actions"
            action_mask[:, -completion_ids.shape[1]:] = 1

            # ---------------
            # 3) Create samples
            # ---------------
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
            )

            # ---------------
            # 4) Compute needed data: log_probs, values, KL
            #    (some may be placeholders if you rely on later pass)
            # ---------------
            # Actor log_probs
            actor_log_probs = self.actor(
                samples.sequences, samples.num_actions, samples.attention_mask
            )  # shape (1, S)

            # Base (initial) policy log_probs
            base_log_probs = self.initial_model(
                samples.sequences, samples.num_actions, samples.attention_mask
            )  # shape (1, S)

            # Critic values (if critic is provided)
            if self.critic:
                values = self.critic(
                    samples.sequences, 
                    samples.num_actions,
                    samples.attention_mask
                )  # shape (1, S)
            else:
                # If no critic is available, store zero or placeholder
                values = torch.zeros_like(actor_log_probs)

            # Approximate KL
            kl = compute_approx_kl(
                actor_log_probs,
                base_log_probs,
                action_mask=samples.action_mask,
                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
            )

            # ---------------
            # 5) Create Experience object
            # ---------------
            exp = Experience(
                sequences=samples.sequences,
                action_log_probs=actor_log_probs,   # shape (1, S)
                values=values,                     # shape (1, S)
                returns=None,
                advantages=None,
                attention_mask=samples.attention_mask,
                action_mask=samples.action_mask,
                info={
                    # offline dataset reward => store in info
                    "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                    "response_length": samples.response_length,
                    "total_length": samples.total_length,
                    "num_actions": samples.num_actions,
                },
                kl=kl,
            )

            # Move to CPU if your pipeline expects CPU-stored experiences
            exp.to_device("cpu")
            experiences.append(exp)

        # ---------------
        # 6) Let the parent's pipeline do batching / processing
        #    This typically sets up GAE or REINFORCE returns, etc.
        # ---------------
        experiences, rewards = self.process_experiences(experiences)

        # ---------------
        # 7) Re-compute final reward w/ KL penalty, advantage/returns
        #    The parent's pipeline might do it for you, but if not:
        # ---------------
        for exp, raw_rew in zip(experiences, rewards):
            exp = exp.to_device(self.device)

            # final reward = raw_rew - KL * kl_coef
            final_reward = compute_reward(
                raw_rew,
                self.kl_ctl.value,
                exp.kl,
                action_mask=exp.action_mask,
                num_actions=exp.info["num_actions"],
                reward_clip_range=self.strategy.args.reward_clip_range,
            )

            # advantage_estimator logic
            if self.advantage_estimator == "gae":
                exp.advantages, exp.returns = self.get_advantages_and_returns(
                    exp.values,
                    final_reward,
                    exp.action_mask,
                    generate_kwargs.get("gamma", 0.99),
                    generate_kwargs.get("lambd", 0.95),
                )
            elif self.advantage_estimator in ["reinforce", "rloo"]:
                exp.returns = self.get_cumulative_returns(
                    final_reward,
                    exp.action_mask,
                    generate_kwargs.get("gamma", 0.99),
                )
                exp.advantages = exp.returns
            else:
                # no recognized advantage estimator
                raise ValueError(f"Unknown advantage_estimator: {self.advantage_estimator}")

            exp.info["return"] = exp.returns.sum().item()
            exp.to_device("cpu")

        return experiences

    def make_experience_batch(self, batch: dict) -> List[Experience]:
        """
        Example for a batched version if you proceed with batched offline data.
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

            # Here, fill actor_log_probs, values, etc. accordingly
            # ...
            actor_log_probs = torch.zeros(1, input_ids.size(-1))  # placeholder
            values = torch.zeros(1, input_ids.size(-1))           # placeholder

            exp = Experience(
                sequences=input_ids.unsqueeze(0),
                action_log_probs=actor_log_probs,
                values=values,
                returns=None,
                advantages=None,
                attention_mask=attention_mask.unsqueeze(0),
                action_mask=action_mask.unsqueeze(0),
                info={
                    "reward": torch.tensor([reward], dtype=torch.float32),
                    "num_actions": len(completion_ids),
                },
                kl=None,
            )
            experiences.append(exp)

        return experiences
