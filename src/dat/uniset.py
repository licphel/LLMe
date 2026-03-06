import logging
from typing import List, Dict

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
        import torch
        from torch.utils.data import Dataset

        stride = stride or (seq_len // 2)
        samples = []

        for idx, item in enumerate(self.data):
            text = item["text"]

            tokens = tokenizer.encode(text)

            if len(tokens) < seq_len + 1:
                continue

            window_count = 0
            for i in range(0, len(tokens) - seq_len, stride):
                chunk = tokens[i : i + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    input_ids = chunk[:-1]
                    target_ids = chunk[1:]
                    samples.append((input_ids, target_ids))
                    window_count += 1

        class _UnisetDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                input_ids, target_ids = self.samples[idx]
                return (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(target_ids, dtype=torch.long),
                )

        return _UnisetDataset(samples)

    def __add__(self, other: "Uniset") -> "Uniset":
        return Uniset(self.data + other.data)

    def clear(self):

        self.data.clear()
