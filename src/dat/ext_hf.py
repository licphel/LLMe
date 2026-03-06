from .loader import DataLoader
import logging
from typing import Optional, Dict
import datasets

logger = logging.getLogger(__name__)


# HuggingFace dataset loader.
class HuggingFaceFetcher(DataLoader):
    def load_hf(
        self,
        path: str,
        name: Optional[str] = None,
        split: str = "train",
        text_column: str = "text",
        **kwargs,
    ) -> int:
        logger.info(f"Begin loading from HuggingFace: {path}")

        try:
            dataset = datasets.load_dataset(path, name=name, split=split, **kwargs)

            count = 0
            for item in dataset:
                text = self._extract_text(item, text_column)
                if text:
                    self.append_text(
                        {
                            "text": text,
                            "source": f"hf://{path}/{split}:{count}",
                            "type": "sft",
                            "metadata": {
                                k: v for k, v in item.items() if k != text_column
                            },
                        }
                    )
                    count += 1

            logger.info(f"{count} samples loaded")
            return count

        except Exception as e:
            logger.error(f"Load fault: {e}")
            return 0

    def _extract_text(self, item: Dict, text_column: str) -> Optional[str]:
        if text_column in item:
            return str(item[text_column])

        for col in ["text", "content", "sentence", "input", "instruction", "output"]:
            if col in item:
                return str(item[col])

        if isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str) and len(value) > 50:
                    return value

        return None
