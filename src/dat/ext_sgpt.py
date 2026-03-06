from .loader import DataLoader
from pathlib import Path
import logging
from .util import readjson

logger = logging.getLogger(__name__)


# ShareGPT .json data loader.
# TODO: Dialog support
class ShareGPTLoader(DataLoader):
    def _collect(self, path: Path) -> int:
        try:
            data = readjson(path)

            if isinstance(data, list):
                count = 0
                for idx, item in enumerate(data):
                    count += self._parse_sharegpt_item(item, path, idx)
                return count
            else:
                return self._parse_sharegpt_item(data, path, 0)

        except Exception as ex:
            logger.error(f"invalid json: {ex}")
            return 0

    def _parse_sharegpt_item(self, item: dict, path: Path, idx: int) -> int:
        try:
            if "conversations" not in item:
                return 0

            formatted = ""
            turn_count = 0

            if "system" in item and item["system"]:
                formatted += f"[System] {item['system']}\n"

            for conv in item["conversations"]:
                role = conv.get("from", "")
                value = conv.get("value", "")

                if role == "human":
                    formatted += f"[User] {value}\n"
                elif role == "gpt":
                    formatted += f"[Assistant] {value}\n"
                    turn_count += 1
                elif role == "system":
                    formatted += f"[System] {value}\n"

            if turn_count > 0:
                formatted += "[Assistant]"
                self.append_text(
                    {
                        "text": formatted,
                        "source": f"{path.name}:{idx}",
                        "type": "sft",
                        "turns": turn_count,
                    }
                )
                return turn_count

        except Exception as e:
            logger.warning(f"fail to pass ShareGPT item: {e}")

        return 0
