import torch
import logging
from typing import Dict, Any
from lib.basepath import Basepath
from dat import Uniset
from tokenizer import Tokenizer
from model import LanguageModel
from trainer import Trainer
from lib import Config
import json
import load as load
from pathlib import Path
from torch.utils.data import DataLoader as TorchDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_current_model = None
_current_tokenizer = None
_current_model_name = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model_dir = Basepath / "models"


def train(model_name: str) -> Dict[str, Any]:
    print(f"\n{'='*50}")
    print(f"Training starts: {model_name}")
    print(f"{'='*50}")

    print("\nLoading datasets...")
    uniset: Uniset = load.get_data_cache()

    if len(uniset.data) == 0:
        raise RuntimeError("No data in 'data/'")

    print(f"\nStats:")
    print(f"  Samples: {len(uniset.data)}")

    # 2. train tokenizer
    print("\nTraining tokenizer...")
    tokenizer = Tokenizer()

    corpus_parts = []
    dialog_count = 0
    text_count = 0

    for item in uniset.data:
        if "dialog" in item:
            dialog_count += 1
            for turn in item["dialog"]:
                if "content" in turn:
                    if turn.get("role") == "user":
                        corpus_parts.append("<user>")
                    elif turn.get("role") == "assistant":
                        corpus_parts.append("<assistant>")
                    elif turn.get("role") == "system":
                        corpus_parts.append("<system>")
                    corpus_parts.append(turn["content"])
            corpus_parts.append("<eos>")
        elif "text" in item:
            text_count += 1
            corpus_parts.append(item["text"])
            corpus_parts.append("<eos>")

    print(f"  Data composition: {dialog_count} dialogs, {text_count} texts")

    if not corpus_parts:
        raise RuntimeError("No valid text content found in data")

    corpus = "\n".join(corpus_parts)
    tokenizer.train(corpus)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # 3. create training dataset
    print("\nCreating training dataset...")
    dataset = uniset.to_torch_dataset(tokenizer, seq_len=Config["max_sequence_length"])

    train_loader = TorchDataLoader(
        dataset,
        batch_size=Config["batch_size"],
        shuffle=True,
        num_workers=2,
        persistent_workers=False,
    )

    # 4. create model
    print("\nCreating model...")
    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        dim=Config["dimensions"],
        n_layers=Config["layers"],
        n_heads=Config["heads"],
        max_seq_len=Config["max_sequence_length"],
    )
    argc = model.count_parameters()
    print(f"  ArgCount: {argc:,}")

    # 5. training section
    print("\nStart training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        lr=Config["learning_rate"],
        device=_device,
    )

    save_dir = _model_dir / model_name
    trainer.train(epochs=Config["epochs"], save_dir=save_dir)

    # 6. save tokenizer and Config
    print("\nSaving model...")
    tokenizer.save(save_dir / "tokenizer.json")
    with open(save_dir / "training_config.json", "w") as f:
        json.dump(Config, f, indent=2)

    # 7. switch to new model
    global _current_model, _current_tokenizer, _current_model_name
    _current_model = model
    _current_tokenizer = tokenizer
    _current_model_name = model_name

    print(f"\n{'='*50}")
    print(f"{model_name} training done！")
    print(f"  Saved at: {save_dir}")
    print(f"{'='*50}")

    return {
        "model_name": model_name,
        "samples": len(uniset.data),
        "vocab_size": tokenizer.vocab_size,
        "parameters": argc,
    }


def resume_train(model_name: str, additional_epochs: int):
    print(f"\n{'='*50}")
    print(f"Resuming training: {model_name}")
    print(f"{'='*50}")

    # 1. check model
    model_dir = _model_dir / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_name}")

    # 2. load cfg
    config_path = model_dir / "training_config.json"
    with open(config_path, "r") as f:
        model_config = json.load(f)

    # 3. load tokenizer
    tokenizer = Tokenizer()
    tokenizer.load(model_dir / "tokenizer.json")

    # 4. recreate model
    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        dim=model_config.get("dimensions", Config["dimensions"]),
        n_layers=model_config.get("layers", Config["layers"]),
        n_heads=model_config.get("heads", Config["heads"]),
        max_seq_len=model_config.get(
            "max_sequence_length", Config["max_sequence_length"]
        ),
    )

    # 5. load pts
    checkpoint_path: Path = Path()

    interrupted_files = list(model_dir.glob("interrupted_epoch_*.pt"))
    if interrupted_files:
        # find newest
        checkpoint_path = max(interrupted_files, key=lambda p: p.stat().st_mtime)
    elif (model_dir / "final.pt").exists():
        checkpoint_path = model_dir / "final.pt"
    elif (model_dir / "best.pt").exists():
        checkpoint_path = model_dir / "best.pt"

    print(f"   Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=_device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        completed_epochs = checkpoint.get("epoch", 0)
        completed_steps = checkpoint.get("step", 0)
        print(
            f"   Previously completed: {completed_epochs} epochs, {completed_steps} steps"
        )
    else:
        model.load_state_dict(checkpoint)
        completed_epochs = 0
        completed_steps = 0

    model.to(_device)

    # 6. load datas
    uniset = load.get_data_cache()
    if len(uniset.data) == 0:
        raise Exception("No data in cache.")

    print(f"\nData stats:")
    print(f"  Samples: {len(uniset.data)}")

    # 7. recreate dataset and loader
    print("\nCreating training dataset...")
    dataset = uniset.to_torch_dataset(
        tokenizer, seq_len=model_config["max_sequence_length"]
    )

    train_loader = TorchDataLoader(
        dataset,
        batch_size=model_config["batch_size"],
        shuffle=True,
        num_workers=2,
        persistent_workers=False,
    )

    if additional_epochs <= 0:
        target_epochs = completed_epochs + 1
    else:
        target_epochs = completed_epochs + additional_epochs

    print(f"\nTraining plan:")
    print(f"   Completed: {completed_epochs} epochs")
    print(f"   Target: {target_epochs} epochs")
    print(f"   Additional: {target_epochs - completed_epochs} epochs")

    # 9. create trainer
    print("\nResuming training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        lr=model_config["learning_rate"],
        device=_device,
    )

    # load optimizer
    if "optimizer_state_dict" in checkpoint:
        # do not load this
        # it will overwrite learning
        # trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.step = completed_steps
        trainer.epoch = completed_epochs
        trainer.losses = checkpoint.get("losses", [])
        print(f"   Loaded optimizer state, resuming from step {trainer.step}")

    # 10. resume train
    trainer.train(
        epochs=target_epochs, save_dir=model_dir, resume_from=completed_epochs
    )

    # 11. update epochs
    model_config["epochs"] = target_epochs
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    # 12. switch to new model
    global _current_model, _current_tokenizer, _current_model_name
    _current_model = model
    _current_tokenizer = tokenizer
    _current_model_name = model_name

    print(f"\n{'='*50}")
    print(f"Resume training completed!")
    print(f"  Model: {model_name}")
    print(f"  Total epochs: {target_epochs}")
    print(f"{'='*50}")

    return {
        "model_name": model_name,
        "total_epochs": target_epochs,
        "completed_epochs": completed_epochs,
        "additional_epochs": target_epochs - completed_epochs,
        "final_loss": trainer.losses[-1] if trainer.losses else 0,
    }


def switch(model_name):
    global _current_model, _current_tokenizer, _current_model_name

    model_dir = Basepath / "models" / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_name}")

    tokenizer = Tokenizer()
    tokenizer.load(model_dir / "tokenizer.json")

    with open(model_dir / "training_config.json", "r") as f:
        Config = json.load(f)

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        dim=Config["dimensions"],
        n_layers=Config["layers"],
        n_heads=Config["heads"],
        max_seq_len=Config["max_sequence_length"],
    )

    checkpoint = torch.load(model_dir / "best.pt", map_location=_device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(_device)
    model.eval()

    _current_model = model
    _current_tokenizer = tokenizer
    _current_model_name = model_name

    print(f"Loaded model: {model_name}")
    return model


def chat(prompt):
    global _current_model, _current_tokenizer
    if _current_model is None:
        print("No model is bound.")
        return

    input_text = f"<user>{prompt}<assistant>"
    response = _generate_text(_current_model, _current_tokenizer, input_text)

    print(response)
    return response


def list_models():
    if not _model_dir.exists():
        return []

    return [d.name for d in _model_dir.iterdir() if d.is_dir()]


# utility method to generate text
def _generate_text(model, tokenizer, prompt):
    model.to(_device)
    model.eval()

    input_text = f"<user>{prompt}<assistant>"
    input_ids = tokenizer.encode(input_text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(_device)

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=Config["max_tokens"],
            temperature=Config["temperature"],
            top_k=Config["top_k"],
            top_p=Config["top_p"],
            repetition_penalty=Config["repetition_penalty"],
            eos_token_id=tokenizer.eos_id,
        )

    full_output = tokenizer.decode(output_ids[0].tolist())

    if "<assistant>" in full_output:
        response = full_output.split("<assistant>")[-1]
        response = response.replace("<eos>", "").strip()
        return response
    return full_output
