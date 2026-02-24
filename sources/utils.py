import torch

def generate_text(model, tokenizer, prompt, max_length=1000, temperature=0.8, device='cpu'):
    model.to(device)
    model.eval()
    
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_length,
            temperature=temperature
        )
    
    return tokenizer.decode(output_ids[0].tolist())