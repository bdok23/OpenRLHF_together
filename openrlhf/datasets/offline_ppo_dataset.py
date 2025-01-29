import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Union

class OfflinePPODataset(Dataset):
    """
    Offline dataset for PPO that stores:
       - prompts
       - completions
       - external reward scores
    """
    def __init__(
        self, 
        data: List[Dict[str, Union[str, float]]],
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int,
        max_completion_length: int
    ):
        """
        data: a list of dicts or tuples containing
              "prompt", "completion", and "reward"
        tokenizer: the tokenizer to use for tokenizing prompts and completions
        max_prompt_length: the maximum length of the tokenized prompt
        max_completion_length: the maximum length of the tokenized completion
        """
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a single record (prompt, completion, reward).
        Structure your item so that your experience maker
        can interpret or directly embed it into an Experience object.
        """
        entry = self.data[idx]
        
        # Tokenize prompt and completion separately
        prompt = self.tokenizer(
            entry["prompt"],
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False
        ).input_ids
        
        completion = self.tokenizer(
            entry["completion"],
            max_length=self.max_completion_length,
            truncation=True,
            add_special_tokens=False
        ).input_ids
        
        return {
            "prompt_ids": prompt,
            "completion_ids": completion,
            "reward": torch.tensor(entry["reward"], dtype=torch.float32)
        }

    def collate_fn(self, batch):
        # Pad sequences
        prompt_ids = [torch.tensor(item["prompt_ids"]) for item in batch]
        completion_ids = [torch.tensor(item["completion_ids"]) for item in batch]
        rewards = torch.stack([item["reward"] for item in batch])
        
        return {
            "prompt_ids": torch.nn.utils.rnn.pad_sequence(
                prompt_ids, 
                batch_first=True, 
                padding_value=self.tokenizer.pad_token_id
            ),
            "completion_ids": torch.nn.utils.rnn.pad_sequence(
                completion_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            ),
            "reward": rewards
        }