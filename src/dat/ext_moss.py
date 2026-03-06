from .loader import DataLoader
from pathlib import Path
from .util import readjson
import logging

logger = logging.getLogger(__name__)


# MOSS .json data loader.
# TODO: Dialog support
class MossLoader(DataLoader):
    def _collect(self, path: Path) -> int:
        try:
            data = readjson(path)

            if isinstance(data, list):
                count = 0
                for item in data:
                    count += self._parse_moss_item(item, path)
                return count
            else:
                return self._parse_moss_item(data, path)

        except Exception as ex:
            logger.error(f"invalid json: {ex}")
            return 0

    def _parse_moss_item(self, item: dict, path: Path) -> int:
        try:
            if "conversation" not in item:
                logger.warning(f"invalid item: miss 'conversation'")
                return 0

            conversation = item["conversation"]
            formatted_dialogue = ""
            turn_count = 0

            for turn_key in sorted(conversation.keys()):
                if not turn_key.startswith("turn_"):
                    continue

                turn = conversation[turn_key]

                if "Human" in turn:
                    formatted_dialogue += f"[User] {turn['Human']}\n"
                if "MOSS" in turn:
                    formatted_dialogue += f"[Assistant] {turn['MOSS']}\n"
                    turn_count += 1

            if turn_count > 0:
                formatted_dialogue += "[Assistant]"

                self.append_text(
                    {
                        "text": formatted_dialogue,
                        "source": f"{path.name}:{item.get('conversation_id', 'unknown')}",
                        "type": "sft",
                        "turns": turn_count,
                        "category": item.get("category", "unknown"),
                    }
                )

                return turn_count

        except Exception as e:
            logger.warning(f"fail to decode MOSS item: {e}")

        return 0
