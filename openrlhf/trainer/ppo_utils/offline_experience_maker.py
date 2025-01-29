import torch
from copy import deepcopy
from .experience_maker import Experience, Samples
from .experience_maker import NaiveExperienceMaker
from typing import List

class OfflineExperienceMaker(NaiveExperienceMaker):
    """
    An 'offline' version of the Experience Maker that
    reads completions and rewards from the dataset, instead of
    generating them on the fly.
    """
    def __init__(self, *args, offline_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.offline_dataset = offline_dataset
        self.tokenizer = args[0]  # Assuming the first argument is the tokenizer
        self.device = kwargs.get('device', 'cuda')
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
            # 1) Convert prompt+completion into token IDs
            # 2) Convert reward into a torch.Tensor
            # 3) Possibly create "Samples" if you want the same shape as online code

            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            completion_ids = self.tokenizer(completion, return_tensors="pt").input_ids.to("cuda")

            # Concatenate tokens so it looks like [prompt_ids, completion_ids]
            sequences = torch.cat([prompt_ids, completion_ids], dim=1)

            # Build (attention_mask, action_mask) similarly if needed
            # or if you use a simpler approach, you can place placeholders.
            attention_mask = torch.ones_like(sequences).to("cuda")
            action_mask = torch.zeros_like(sequences).to("cuda")
            action_mask[:, -completion_ids.size(1):] = 1

            # Wrap them in a Samples object (mimicking the normal pipeline)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
            )

            # Wrap them in an Experience
            exp = Experience(
                samples,
                reward=torch.tensor([reward], dtype=torch.float32).to("cuda"),
                # For KL penalty, you can store "kl", or keep it None
                kl=None,
            )
            # Also store any needed info, e.g. "response_length", "num_actions", etc.
            exp.info = {
                "reward": exp.reward,
                "num_actions": action_mask.size(1),
                "response_length": exp.samples.response_length,
            }

            # Move data to CPU for consistency
            exp.to_device("cpu")
            experiences.append(exp)

        # Optionally process experiences the same way the parent does
        experiences, rewards = self.process_experiences(experiences)

        # Since we already have the final reward, no kl or shaping is strictly needed
        # but the parent's process_experiences() might do advantage estimation, etc.
        # so let it do the GAE logic, if needed.
        for experience, rew_val in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            # If you want to run advantage calculation:
            #   1) You might supply "experience.values" from a learned critic
            #   2) Then call self.get_advantages_and_returns(...)
            # Otherwise just store the offline advantage
            # (If you have "raw advantage" in your dataset, you can place it here.)
            experience.info["return"] = rew_val.sum().item()
            experience.to_device("cpu")

        return experiences

    def make_experience_batch(self, batch: dict) -> List[Experience]:
        experiences = []
        
        # Process batch
        for prompt_ids, completion_ids, reward in zip(
            batch["prompt_ids"], 
            batch["completion_ids"],
            batch["reward"]
        ):
            # Combine prompt and completion
            input_ids = torch.cat([prompt_ids, completion_ids], dim=-1)
            
            # Create attention mask
            attention_mask = torch.ones_like(input_ids)
            
            # Create action mask (only mask completion part)
            action_mask = torch.zeros_like(input_ids)
            action_mask[len(prompt_ids):] = 1
            
            # Create samples
            samples = Samples(
                sequences=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                action_mask=action_mask.unsqueeze(0),
                num_actions=len(completion_ids),
                packed_seq_lens=None,
                response_length=len(completion_ids),
                total_length=len(input_ids)
            )
            
            # Create experience
            exp = Experience(
                samples=samples,
                reward=reward,
                values=torch.zeros_like(reward),  # Will be filled by critic
                log_probs=torch.zeros_like(input_ids),  # Will be recalculated
                advantages=torch.zeros_like(reward),
                returns=torch.zeros_like(reward),
                kl=None
            )
            
            experiences.append(exp)
        
        return experiences
