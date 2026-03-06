import logging
from pathlib import Path
from .uniset import Uniset

logger = logging.getLogger(__name__)


# processes multi-format dataset into unified training data.
class DataLoader:
    def __init__(self, seqlen: int, stride):
        self.seqlen = seqlen
        # by default stride is seqlen / 2.
        self.stride = stride if stride is not None else (seqlen / 2.0)
        self.uniset = Uniset()

    # loads a dataset.
    def load(self, path: Path):
        if not path.exists():
            raise Exception(f"path '{path}' not exists")
        if path.is_file():
            self._load_file(path)
        else:
            for chpath in path.rglob("*"):
                self._load_file(chpath)

    # PRIVATE: loads a dataset from a single file.
    def _load_file(self, path: Path):
        self._collect(path)

    # PRIVATE: collects raw text.
    def _collect(self, path: Path) -> int:
        return 0

    # appends a raw text entry.
    def append_text(self, raw):
        self.uniset.data.append(raw)
