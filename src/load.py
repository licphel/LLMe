from lib.basepath import Basepath
from datasets import load_dataset
import logging
from pathlib import Path
from dat import Uniset, DataFormat
from lib import TrainCfg

logger = logging.getLogger(__name__)

_data_cache = None


def get_data_cache():
    global _data_cache
    if _data_cache is None:
        _data_cache = Uniset()
    return _data_cache


# loads all local data in data/
def scan(path: str) -> dict:
    cache = get_data_cache()

    dir = Basepath / path
    if not dir.exists():
        raise FileNotFoundError(f"Data directory not found: {dir}")

    format_stats = {}
    total_before = len(cache.data)

    print(f"   Scanning {dir}...")

    for file_path in dir.rglob("*"):
        if file_path.is_dir():
            continue
        
        fmt = _detect(file_path)
        try:
            logger.info(f"Loading {file_path} as {fmt}")
            data = DataFormat.load(
                file_path,
                fmt,
                seqlen=TrainCfg["max_sequence_length"],
                stride=TrainCfg["stride"],
            )
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


def _detect(file_path: Path) -> str:
    if not file_path.is_file():
        return "unknown"

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
            return fmt
        except:
            return "unknown"
    return "unknown"


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
