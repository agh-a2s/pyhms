"""
    Universal stopping conditions. May be used as both local and global s.c.
"""
from abc import ABC, abstractmethod
from typing import Any, Union

from .deme import Deme
from .tree import DemeTree

class usc(ABC):
    @abstractmethod
    def satisfied(self, obj: Union[DemeTree, Deme]) -> bool:
        raise NotImplementedError()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.satisfied(*args, **kwds)
        
class metaepoch_limit(usc):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def satisfied(self, obj: Union[DemeTree, Deme]) -> bool:
        return obj.metaepoch_count >= self.limit

    def __str__(self) -> str:
        return f"metaepoch_limit({self.limit})"

class dont_stop(usc):
    def satisfied(self, _: Union[DemeTree, Deme]) -> bool:
        return False

    def __str__(self) -> str:
        return "dont_stop"

class dont_run(usc):
    def satisfied(self, _: Union[DemeTree, Deme]) -> bool:
        return True

    def __str__(self) -> str:
        return "dont_run"
