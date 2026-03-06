import json
from pathlib import Path

# a char tokenizer.
class Tokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
        self.pad_id = None
        self.unk_id = None
        self.bos_id = None
        self.eos_id = None
        self.user_id = None
        self.assistant_id = None

    def train(self, text):
        chars = sorted(list(set(text)))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

        special_tokens = [
            "<pad>",   # padding
            "<unk>",   # unknown
            "<bos>",   # begin of sentence
            "<eos>",   # end of sentence
            "<user>",  # user mark
            "<assistant>",  # assistant mark
            "<sep>",   # separator
        ]
        
        for token in special_tokens:
            if token not in self.stoi:
                self.stoi[token] = self.vocab_size
                self.itos[self.vocab_size] = token
                self.vocab_size += 1

        self.pad_id = self.stoi.get("<pad>")
        self.unk_id = self.stoi.get("<unk>")
        self.bos_id = self.stoi.get("<bos>")
        self.eos_id = self.stoi.get("<eos>")
        self.user_id = self.stoi.get("<user>")
        self.assistant_id = self.stoi.get("<assistant>")

        print(f"Tokenizer vocab size: {self.vocab_size}")
        print(f"Special tokens: pad={self.pad_id}, eos={self.eos_id}, user={self.user_id}, assistant={self.assistant_id}")
        return self

    def encode(self, text):
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids):
        text = ""
        for i in ids:
            token = self.itos.get(i, "<unk>")
            if token not in ["<pad>", "<bos>", "<eos>", "<user>", "<assistant>", "<sep>"]:
                text += token
        return text

    def encode_dialog(self, dialog):
        tokens = []
        for turn in dialog:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                tokens.append(self.user_id)
            elif role == "assistant":
                tokens.append(self.assistant_id)
            tokens.extend(self.encode(content))
        # append <eos> for end of a sentence.
        tokens.append(self.eos_id)
        return tokens

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "stoi": self.stoi,
                    "itos": {str(k): v for k, v in self.itos.items()},
                    "vocab_size": self.vocab_size,
                    "pad_id": self.pad_id,
                    "eos_id": self.eos_id,
                    "user_id": self.user_id,
                    "assistant_id": self.assistant_id,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.stoi = data["stoi"]
            self.itos = {int(k): v for k, v in data["itos"].items()}
            self.vocab_size = data["vocab_size"]
            self.pad_id = data.get("pad_id")
            self.eos_id = data.get("eos_id")
            self.user_id = data.get("user_id")
            self.assistant_id = data.get("assistant_id")
        return self
