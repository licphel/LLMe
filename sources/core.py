import torch
from pathlib import Path
from model import MiniLM
from tokenizer import CharTokenizer
from utils import generate_text

_current_model = None
_current_tokenizer = None
_current_model_name = None
_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def switch(model_name):
    global _current_model, _current_tokenizer, _current_model_name
    
    model_dir = Path(f"models/{model_name}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model {model_name} not found")
    
    tokenizer = CharTokenizer()
    tokenizer.load(model_dir / 'tokenizer.json')
    
    import json
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)

    valid_params = ['vocab_size', 'dim', 'n_layers', 'n_heads', 'max_seq_len', 'dropout']
    filtered_config = {k: v for k, v in config.items() if k in valid_params}
    
    model = MiniLM(**filtered_config)
    checkpoint = torch.load(model_dir / 'best.pt', map_location=_device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                          if not k.endswith('.causal_mask')}
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(_device)
    model.eval()
    
    _current_model = model
    _current_tokenizer = tokenizer
    _current_model_name = model_name
    
    return model

def chat(prompt):
    global _current_model, _current_tokenizer
    
    if _current_model is None:
        try:
           switch("llme")
        except Exception:
            print("No model saved")
            return

    response = generate_text(
        _current_model,
        _current_tokenizer,
        prompt,
        max_length=100,
        temperature=0.8,
        device=_device
    )
    
    print(response)
    return response