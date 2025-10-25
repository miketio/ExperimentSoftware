# xyzstage_base.py
from abc import ABC, abstractmethod

class XYZStageBase(ABC):
    """
    Abstract base class describing the API for a 3D XYZ stage.

    All positions/units are in nanometers (nm) to match your existing code.
    Subclasses must implement the concrete hardware (or mock) behavior.
    """

    @abstractmethod
    def move_abs(self, axis: str, pos: int) -> None:
        """Move the given axis ('x','y','z') to absolute position `pos` (nm)."""
        raise NotImplementedError

    @abstractmethod
    def move_rel(self, axis: str, shift: int, sleep_time: float = 0.01) -> None:
        """Move the given axis by relative amount `shift` (nm)."""
        raise NotImplementedError

    @abstractmethod
    def get_pos(self, axis: str) -> int:
        """Return current position of axis (nm)."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Cleanly close the connection / free resources."""
        raise NotImplementedError

    # Context manager convenience
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except Exception:
            # don't mask original exceptions
            pass
