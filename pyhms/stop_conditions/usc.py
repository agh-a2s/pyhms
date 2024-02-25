from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from pyhms.demes.abstract_deme import AbstractDeme
    from pyhms.tree import DemeTree


class UniversalStopCondition(ABC):
    @abstractmethod
    def __call__(self, obj: Union["DemeTree", "AbstractDeme"]) -> bool:
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.__class__.__name__


class MetaepochLimit(UniversalStopCondition):
    def __init__(self, limit: int) -> None:
        self.limit = limit

    def __call__(self, obj: Union["DemeTree", "AbstractDeme"]) -> bool:
        return obj.metaepoch_count >= self.limit

    def __str__(self) -> str:
        return f"MetaepochLimit({self.limit})"


class DontStop(UniversalStopCondition):
    def __call__(self, _: Union["DemeTree", "AbstractDeme"]) -> bool:
        return False


class DontRun(UniversalStopCondition):
    def __call__(self, _: Union["DemeTree", "AbstractDeme"]) -> bool:
        return True
