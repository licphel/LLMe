import logging
from pathlib import Path
from typing import Type, Optional

from ._hugging_face import HuggingFaceFetcher
from ._alpaca import AlpacaLoader
from ._moss import MossLoader
from ._share_gpt import ShareGPTLoader
from ._txt import TextLoader
from .loader import DataLoader
from .uniset import Uniset

logger = logging.getLogger(__name__)


# data format supports.
class DataFormat:
    TXT = "txt"
    MOSS = "moss"
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    OPENAI = "openai"
    HF = "hf"

    # gets a loader class from the given format.
    @staticmethod
    def loaderclass(fmt: str) -> Type[DataLoader]:
        if fmt == DataFormat.TXT:
            return TextLoader
        if fmt == DataFormat.MOSS:
            return MossLoader
        if fmt == DataFormat.SHAREGPT:
            return ShareGPTLoader
        if fmt == DataFormat.ALPACA:
            return AlpacaLoader
        if fmt == DataFormat.HF:
            return HuggingFaceFetcher
        raise Exception(f"unknown format: {fmt}")

    # loads a dataset with a path.
    @staticmethod
    def load(path: Path, fmt: str, seqlen: int = 256, stride=None) -> Uniset:
        lclass: Type[DataLoader] = DataFormat.loaderclass(fmt)
        loader: DataLoader = lclass(seqlen, stride)
        loader.load(path)
        return loader.uniset

    # loads a dataset from HuggingFace.
    @staticmethod
    def load_huggingface(
            dataset_path: str,
            name: Optional[str] = None,
            split: str = "train",
            seqlen: int = 256,
            stride=None,
            **kwargs,
    ) -> Uniset:
        loader = HuggingFaceFetcher(seqlen, stride)
        loader.load_hf(dataset_path, name=name, split=split, **kwargs)
        return loader.uniset
