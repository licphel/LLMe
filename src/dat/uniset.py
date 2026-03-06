import logging
from typing import List, Dict
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# a unified training data object.
class Uniset:
    def __init__(self, data: List[Dict] = []):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def append(self, item: Dict):
        self.data.append(item)

    def extend(self, items: List[Dict]):
        self.data.extend(items)

    def to_torch_dataset(self, tokenizer, seq_len: int = 256, stride=None):
        stride = stride or (seq_len // 2)
        samples = []

        for idx, item in enumerate(self.data):
            # check if it is dialog structured.
            if "dialog" in item:
                tokens = tokenizer.encode_dialog(item["dialog"])
            else:
                # pure text
                text = item["text"]
                tokens = tokenizer.encode(text)
                # add <eos>
                tokens.append(tokenizer.eos_id)

            if len(tokens) < seq_len + 1:
                # too short, fill it to seq_len + 1
                tokens = tokens + [tokenizer.pad_id] * (seq_len + 1 - len(tokens))

            # sampling
            for i in range(0, len(tokens) - seq_len, stride):
                chunk = tokens[i : i + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    input_ids = chunk[:-1]
                    target_ids = chunk[1:]
                    loss_mask = self._create_loss_mask(
                        tokens[i : i + seq_len], tokenizer
                    )

                    samples.append((input_ids, target_ids, loss_mask))

        return _UnisetDataset(samples)

    def _create_loss_mask(self, tokens, tokenizer):
        mask = [0] * len(tokens)
        in_assistant = False

        for i, token in enumerate(tokens):
            if token == tokenizer.assistant_id:
                in_assistant = True
                continue
            elif token == tokenizer.user_id or token == tokenizer.eos_id:
                in_assistant = False

            if in_assistant:
                mask[i] = 1

        return mask

    def __add__(self, other: "Uniset") -> "Uniset":
        return Uniset(self.data + other.data)

    def clear(self):
        self.data.clear()


class _UnisetDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if len(item) == 3:
            input_ids, target_ids, loss_mask = item
            return (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_ids, dtype=torch.long),
                torch.tensor(loss_mask, dtype=torch.bool),
            )
        else:
            input_ids, target_ids = item
            return (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_ids, dtype=torch.long),
            )
