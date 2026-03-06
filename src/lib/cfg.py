from typing import Dict
import logging
import json
from pathlib import Path
from lib.basepath import Basepath

logger = logging.getLogger(__name__)
_cfg_path = Path()

# global config json object
Config: Dict = {}

# config get or create.
_cfg_path = Basepath / "config/settings.json"
try:
    with open(_cfg_path, "r", encoding="utf-8") as file:
        Config = json.load(file)
except Exception as ex:
    Config = {
        "max_sequence_length": 256,
        "stride": 128,
        "dimensions": 512,
        "layers": 8,
        "heads": 8,
        "learning_rate": 0.0003,
        "epochs": 50,
        "batch_size": 32,
        "max_length": 1000,
        "temperature": 0.08
    }
    with open(_cfg_path, "w", encoding="utf-8") as f:
        json.dump(Config, f, indent=2)

logger.debug(f"load config/settings.json: {Config}")

# expose config object.
Config = Config
