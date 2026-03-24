from .dense import DenseModel
from .moe import MoEModel
from typing import Union

def byname(name) -> Union[type[DenseModel], type[MoEModel]]:
  if name == "moe":
    return MoEModel
  if name == "dense":
    return DenseModel
  raise Exception(f"No such arch: {name}")

__all__ = [
  "DenseModel",
  "MoEModel",
  "byname"
]
