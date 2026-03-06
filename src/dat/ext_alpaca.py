from .loader import DataLoader
from pathlib import Path
from .util import readjson
import logging

logger = logging.getLogger(__name__)


# Alpaca .json data loader.
class AlpacaLoader(DataLoader):
    def _collect(self, path: Path) -> int:
        try:
            data = readjson(path)

            if isinstance(data, list):
                count = 0
                for idx, item in enumerate(data):
                    if self._parse_alpaca_item(item, path, idx):
                        count += 1
                return count
            else:
                return 1 if self._parse_alpaca_item(data, path, 0) else 0

        except Exception as ex:
            logger.error(f"invalid json: {ex}")
            return 0

    def _parse_alpaca_item(self, item: dict, path: Path, idx: int) -> bool:
        try:
            if "instruction" not in item or "output" not in item:
                return False

            dialog = []

            if "system" in item and item["system"]:
                dialog.append({"role": "system", "content": item["system"]})

            instruction = item["instruction"]
            input_text = item.get("input", "")

            if input_text:
                user_content = f"{instruction}\n{input_text}"
            else:
                user_content = instruction

            dialog.append({"role": "user", "content": user_content})
            dialog.append({"role": "assistant", "content": item["output"]})

            self.append_text(
                {"dialog": dialog, "source": f"{path.name}:{idx}", "type": "sft"}
            )

            return True

        except Exception as e:
            logger.warning(f"fail to pass Alpaca item: {e}")
            return False
