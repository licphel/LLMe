from .loader import DataLoader
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


# Alpaca .json data loader.
class AlpacaLoader(DataLoader):
    def _collect(self, text: str, path: Path) -> int:
        try:
            data = json.loads(text)

            if isinstance(data, list):
                count = 0
                for idx, item in enumerate(data):
                    if self._parse_alpaca_item(item, path, idx):
                        count += 1
                return count
            else:
                return 1 if self._parse_alpaca_item(data, path, 0) else 0

        except json.JSONDecodeError:
            logger.error(f"invalid json: {path}")
            return 0

    def _parse_alpaca_item(self, item: dict, path: Path, idx: int) -> bool:
        try:
            if "instruction" not in item or "output" not in item:
                return False

            formatted = ""

            if "system" in item and item["system"]:
                formatted += f"[System] {item['system']}\n"

            instruction = item["instruction"]
            input_text = item.get("input", "")

            if input_text:
                user_content = f"{instruction}\n{input_text}"
            else:
                user_content = instruction

            formatted += f"[User] {user_content}\n"
            formatted += f"[Assistant] {item['output']}\n"
            formatted += "[Assistant]"

            self.append_text(
                {"text": formatted, "source": f"{path.name}:{idx}", "type": "sft"}
            )

            return True

        except Exception as e:
            logger.warning(f"fail to pass Alpaca item: {e}")
            return False
