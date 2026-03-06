import torch
from pathlib import Path
import logging
from typing import Dict, Any

from dat import DataFormat, Uniset, HFLoader
from tokenizer import Tokenizer
from model import LanguageModel
from trainer import Trainer
from config.fetch import Config
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_current_model = None
_current_tokenizer = None
_current_model_name = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_MODELS_DIR = Path("models")
_data_cache = None

def get_data_cache():
    global _data_cache
    if _data_cache is None:
        from dat import Uniset

        _data_cache = Uniset()
    return _data_cache


def load_local_data(data_dir: str = "data") -> dict:
    cache = get_data_cache()
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    format_stats = {}
    total_before = len(cache.data)

    print(f"   Scanning {data_path}...")

    for file_path in data_path.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            fmt = "txt"
        elif suffix in [".json", ".jsonl"]:
            try:
                import json

                with open(file_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    data = json.loads(first_line)

                    if isinstance(data, dict):
                        if "conversation" in data:
                            fmt = "moss"
                        elif "conversations" in data:
                            fmt = "sharegpt"
                        elif "instruction" in data:
                            fmt = "alpaca"
                        else:
                            fmt = "json"
                    else:
                        fmt = "json"
            except:
                fmt = "unknown"
        else:
            continue

        try:
            logger.info(f"Loading {file_path} as {fmt}")
            data = DataFormat.load(file_path, fmt, seqlen=256)
            cache += data
            format_stats[fmt] = format_stats.get(fmt, 0) + len(data.data)
        except Exception as e:
            logger.warning(f"Skip {file_path}: {e}")

    total_after = len(cache.data)

    return {
        "total": total_after,
        "added": total_after - total_before,
        "by_format": format_stats,
    }


def load_hf_data(
    dataset_name: str, split: str = "train", text_column: str = ""
) -> dict:
    cache = get_data_cache()
    
    try:
        from dat.ext_hf import HFLoader

        loader = HFLoader(Config["max_sequence_length"], Config["stride"])

        if text_column is None:
            test_loader = HFLoader(seqlen=256)
            test_loader.load_hf(dataset_name, split=split, limit=5)
            if len(test_loader.uniset.data) > 0:
                sample = test_loader.uniset.data[0]
                for col in ["text", "content", "output", "instruction", "sentence"]:
                    if col in sample.get("metadata", {}):
                        text_column = col
                        break

            if text_column is None:
                text_column = "text"

        count = loader.load_hf(path=dataset_name, split=split, text_column=text_column)

        cache += loader.uniset

        return {
            "samples": count,
            "text_column": text_column,
            "total": len(cache.data),
            "added": len(loader.uniset.data),
        }
    except Exception as e:
        logger.error(f"Failed to load HF data: {e}")
        raise


def clear_data_cache():
    global _data_cache
    _data_cache = None
    import gc

    gc.collect()


def get_data_stats() -> dict:
    cache = get_data_cache()

    if len(cache.data) == 0:
        return {"total": 0, "by_source": {}, "by_type": {}}

    by_source = {}
    by_type = {}

    for item in cache.data:
        source = item.get("source", "unknown").split(":")[0]
        by_source[source] = by_source.get(source, 0) + 1

        typ = item.get("type", "unknown")
        by_type[typ] = by_type.get(typ, 0) + 1

    result = {"total": len(cache.data), "by_source": by_source, "by_type": by_type}

    if len(cache.data) > 0:
        first = cache.data[0]["text"]
        result["sample_preview"] = first[:100] + "..." if len(first) > 100 else first

    return result


def train(model_name: str) -> Dict[str, Any]:
    print(f"\n{'='*50}")
    print(f"Training starts: {model_name}")
    print(f"{'='*50}")

    print("\nLoading datasets...")
    uniset: Uniset = get_data_cache()

    if len(uniset.data) == 0:
        raise RuntimeError("No data in 'data/'")

    print(f"\nStats:")
    print(f"  Samples: {len(uniset.data)}")
    
    # 2. train tokenizer
    print("\nTraining tokenizer...")
    tokenizer = Tokenizer()

    all_texts = [item["text"] for item in uniset.data]
    corpus = "\n".join(all_texts)
    tokenizer.train(corpus)
    print(f"  VocabSize: {tokenizer.vocab_size}")

    # 3. create training dataset
    print("\nCreating training dataset...")
    dataset = uniset.to_torch_dataset(tokenizer, seq_len=Config["max_sequence_length"])
    from torch.utils.data import DataLoader as TorchDataLoader

    train_loader = TorchDataLoader(
        dataset, batch_size=Config["batch_size"], shuffle=True, num_workers=0
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

    save_dir = _MODELS_DIR / model_name
    trainer.train(epochs=Config["epochs"], save_dir=str(save_dir))

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
        "parameters": argc
    }


def load_model(model_name):
    global _current_model, _current_tokenizer, _current_model_name

    model_dir = _MODELS_DIR / model_name
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

    response = generate_text(_current_model, _current_tokenizer, prompt)

    print(response)
    return response


def switch(model_name):
    load_model(model_name)


def list_models():
    if not _MODELS_DIR.exists():
        return []

    return [d.name for d in _MODELS_DIR.iterdir() if d.is_dir()]


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=Config["max_length"],
    temperature=Config["temperature"]
):
    model.to(_device)
    model.eval()

    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(_device)

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor, max_new_tokens=max_length, temperature=temperature
        )

    return tokenizer.decode(output_ids[0].tolist())
