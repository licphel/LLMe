from .loader import DataLoader
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# .txt data loader.
class TextLoader(DataLoader):
    def _collect(self, text: str, path: Path) -> int:
        self.append_text({"text": text, "source": str(path), "type": "pretrain"})
        return 1
