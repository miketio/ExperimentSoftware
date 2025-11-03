# mock_stage.py
from HardwareControl.SetupMotor.xyzStageBase import XYZStageBase
import time

class MockXYZStage(XYZStageBase):
    """
    Simple in-memory mock stage. Positions are stored as ints (nm).
    Useful for unit tests and local development without hardware.
    """

    def __init__(self, start_positions=None):
        self._pos = {'x': 0, 'y': 0, 'z': 0}
        if start_positions:
            self._pos.update(start_positions)
        self._closed = False

    def move_abs(self, axis: str, pos: int) -> None:
        if axis not in self._pos:
            raise ValueError(axis)
        # simulate movement time proportional to distance (very small)
        distance = abs(self._pos[axis] - pos)
        time.sleep(min(0.01 + distance / 1e7, 0.1))
        self._pos[axis] = int(pos)

    def move_rel(self, axis: str, shift: int, sleep_time: float = 0.01) -> None:
        if axis not in self._pos:
            raise ValueError(axis)
        self._pos[axis] += int(shift)
        time.sleep(sleep_time)

    def get_pos(self, axis: str) -> int:
        if axis not in self._pos:
            raise ValueError(axis)
        return int(self._pos[axis])

    def close(self) -> None:
        self._closed = True
