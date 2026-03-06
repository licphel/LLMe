from .loader import DataLoader
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# .txt data loader.
class TextLoader(DataLoader):
    def _collect(self, path: Path) -> int:
        with open(path, "r", encoding="utf-8") as f:
            self.append_text({"text": f.read(), "source": str(path), "type": "pretrain"})
        return 1
