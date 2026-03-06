import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

logger = logging.getLogger(__name__)


def readjson(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return _load_jsonl(path)
    elif suffix == ".json":
        return _load_json(path)
    else:
        raise Exception(f"unknown format: {path}")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    result = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                result.append(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num} is not valid JSON: {e}")
                result.append({"_raw_text": line, "_line": line_num, "_error": str(e)})

    logger.info(f"Loaded {len(result)} items from {path} (JSONL)")
    return result


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        result = data
    elif isinstance(data, dict):
        result = [data]
    else:
        result = [{"_data": data}]

    logger.info(f"Loaded {len(result)} items from {path} (JSON)")
    return result
