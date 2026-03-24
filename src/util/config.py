import json
import logging
from pathlib import Path
from typing import Dict
import os
from util.basepath import Basepath

logger: logging.Logger = logging.getLogger(__name__)

# -----------------------------
DEFAULT_TRAIN_CFG: Dict = {
    "architecture": "moe",
    "max_sequence_length": 256,
    "stride": 128,
    "dimensions": 256,
    "layers": 6,
    "heads": 8,
    "learning_rate": 2e-4,
    "epochs": 12,
    "batch_size": 8,
    "dropout": 0.1
}

DEFAULT_ARG_CFG: Dict = {
    "max_tokens": 80,
    "temperature": 0.75,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.15
}

# global config json objects
TRAIN_CFG: Dict = {}
ARG_CFG: Dict = {}
# -----------------------------

# config get or create.
_cfg_path: Path = Basepath / "configs"
if not _cfg_path.exists():
  os.makedirs(_cfg_path)

# train.json
try:
    with open(_cfg_path / "train.json", "r", encoding="utf-8") as file:
        TRAIN_CFG = json.load(file)
except IOError as ex:
    TRAIN_CFG = DEFAULT_TRAIN_CFG

    with open(_cfg_path / "train.json", "w", encoding="utf-8") as f:
        json.dump(TRAIN_CFG, f, indent=2)

# args.json
try:
    with open(_cfg_path / "args.json", "r", encoding="utf-8") as file:
        ARG_CFG = json.load(file)
except IOError as ex:
    ARG_CFG = DEFAULT_ARG_CFG

    with open(_cfg_path / "args.json", "w", encoding="utf-8") as f:
        json.dump(ARG_CFG, f, indent=2)
