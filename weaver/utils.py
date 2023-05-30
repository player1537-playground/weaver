"""

"""

from __future__ import annotations
from os import PathLike
from typing import NewType, Union, BinaryIO

__all__ = [
    'FileLike',
]


FileLike = NewType('FileLike', Union[BinaryIO, PathLike, str])