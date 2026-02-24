import argparse
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader

from tokenizer import CharTokenizer
from model import MiniLM
from trainer import Trainer, TextDataset

def main():
    parser = argparse.ArgumentParser(description='Training args')
    parser.add_argument('--data', '-d', required=True, help='Dataset folder')
    parser.add_argument('--name', '-n', default='my_model', help='Model serial name')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Epoches')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-len', '-s', type=int, default=128, help='Sequence length')
    parser.add_argument('--dim', type=int, default=256, help='Dimensions')
    parser.add_argument('--layers', '-l', type=int, default=4, help='Transformer layer count')
    parser.add_argument('--heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='auto', help='Device used')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("\n" + "="*60)
    print(f" LLMe Training Process")
    print("="*60)
    print(f"Model: {args.name}")
    print(f"On Device: {device}")
    
    data_dir = Path(args.data)
    txt_files = list(data_dir.glob('*.txt'))
    
    if not txt_files:
        print(f"No data found!")
        return
    
    all_text = ""
    for f in txt_files:
        with open(f, 'r', encoding='utf-8') as file:
            text = file.read()
            all_text += text + "\n"
        print(f"   - DAT. {f.name}: {len(text)} chars")
    
    tokenizer = CharTokenizer()
    tokenizer.train(all_text)
    
    dataset = TextDataset(all_text, tokenizer, seq_len=args.seq_len)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    model = MiniLM(
        vocab_size=tokenizer.vocab_size,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        max_seq_len=args.seq_len
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        device=device
    )
    
    trainer.train(epochs=args.epochs, save_dir=f"models/{args.name}")
    
    tokenizer.save(f"models/{args.name}/tokenizer.json")

    config = {
        'vocab_size': tokenizer.vocab_size,
        'dim': args.dim,
        'n_layers': args.layers,
        'n_heads': args.heads,
        'max_seq_len': args.seq_len,
        'model_name': args.name
    }
    config_path = f"models/{args.name}/config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n Model saved.")

if __name__ == "__main__":
    main()