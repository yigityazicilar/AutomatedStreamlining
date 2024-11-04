from typing import Optional, Set, TypeVar
from functools import wraps
from time import time


def get_streamliner_repr_from_set(streamliner_combo: Set[str]) -> str:
    return "-".join(sorted(list(streamliner_combo)))


def get_streamliner_repr_from_str(streamliner_combo: str) -> str:
    return "-".join(sorted(list(streamliner_combo)))


T = TypeVar("T")


def unwrap(opt: Optional[T]) -> T:
    if opt is None:
        raise ValueError("Optional Value is None")
    return opt
