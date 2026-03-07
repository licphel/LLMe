import json
from lib.basepath import Basepath
from datasets import load_dataset
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_fetch_dir = Basepath / "data_fetched"

# fetches a huggingface dataset to ..program/data_fetched/
def fetch_huggingface(dataset_name: str, split: str = "all") -> dict:
    save_dir = _fetch_dir / dataset_name.replace("/", "_")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"   Downloading to: {save_dir}")

    stats = {"samples": 0, "files": 0, "splits": {}}

    try:
        if split == "all":
            dataset = load_dataset(dataset_name)

            for split_name in dataset.keys():
                split_data = dataset[split_name]
                count = _save_hf_split(split_data, save_dir, str(split_name))
                stats["splits"][split_name] = count
                stats["samples"] += count
                stats["files"] += 1
        else:
            dataset = load_dataset(dataset_name, split=split)
            count = _save_hf_split(dataset, save_dir, split)
            stats["splits"][split] = count
            stats["samples"] = count
            stats["files"] = 1

        info = {
            "dataset": dataset_name,
            "splits": list(stats["splits"].keys()),
            "total_samples": stats["samples"],
        }
        with open(save_dir / "dataset_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print(f"   Dataset info saved")

    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

    return stats


def _save_hf_split(dataset, save_dir: Path, split_name: str) -> int:
    output_file = save_dir / f"{split_name}.jsonl"
    count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            text_item = {}
            for key in [
                "text",
                "content",
                "instruction",
                "output",
                "input",
                "conversation",
            ]:
                if key in item:
                    text_item[key] = item[key]

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    print(f"   Saved {split_name}: {count} samples to {output_file}")
    return count