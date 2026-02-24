import json
from pathlib import Path

class CharTokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
    
    def train(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        for token in special_tokens:
            if token not in self.stoi:
                self.stoi[token] = self.vocab_size
                self.itos[self.vocab_size] = token
                self.vocab_size += 1
        
        print(f"Tokenizer vocab size: {self.vocab_size}")
        return self
    
    def encode(self, text):
        return [self.stoi.get(ch, self.stoi.get('<unk>', 0)) for ch in text]
    
    def decode(self, ids):
        return ''.join([self.itos.get(i, '<unk>') for i in ids])
    
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'stoi': self.stoi,
                'itos': {str(k): v for k, v in self.itos.items()},
                'vocab_size': self.vocab_size
            }, f, ensure_ascii=False, indent=2)
    
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.stoi = data['stoi']
            self.itos = {int(k): v for k, v in data['itos'].items()}
            self.vocab_size = data['vocab_size']
        return self