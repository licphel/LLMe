from typing import Dict
import logging
import json
from pathlib import Path
from lib.basepath import Basepath

logger = logging.getLogger(__name__)
_cfg_path = Path()

# global config json objects
TrainCfg: Dict = {}
ArgsCfg: Dict = {}

# config get or create.
_cfg_path = Basepath / "configs"
try:
    with open(_cfg_path / "train.json", "r", encoding="utf-8") as file:
        TrainCfg = json.load(file)
    with open(_cfg_path / "args.json", "r", encoding="utf-8") as file:
        ArgsCfg = json.load(file)
except Exception as ex:
    TrainCfg = {
        "max_sequence_length": 256,
        "stride": 128,
        "dimensions": 256,
        "layers": 6,
        "heads": 8,
        "learning_rate": 2e-4,
        "epochs": 12,
        "batch_size": 8
    }
    
    with open(_cfg_path / "train.json", "w", encoding="utf-8") as f:
        json.dump(TrainCfg, f, indent=2)

try:
    with open(_cfg_path / "args.json", "r", encoding="utf-8") as file:
        ArgsCfg = json.load(file)
except Exception as ex:
    ArgsCfg = {
        "max_tokens": 80,
        "temperature": 0.75,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.15
    }
    
    with open(_cfg_path / "args.json", "w", encoding="utf-8") as f:
        json.dump(ArgsCfg, f, indent=2)
