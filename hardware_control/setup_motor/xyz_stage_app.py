# xyzstage_app.py
from hardware_control.setup_motor.xyz_stage_base import XYZStageBase
from typing import Iterable, Tuple

class XYZStageApp:
    """
    Application-level logic that depends only on XYZStageBase.
    Add high-level procedures here (scan patterns, safe moves, logging, etc).
    """

    def __init__(self, stage: XYZStageBase):
        self.stage = stage
        print("3D-Stage application initialized")

    # Thin wrappers that can add logging, retries, safety checks
    def move_abs(self, axis: str, pos: int):
        print(f"[APP] move_abs {axis} -> {pos} nm")
        self.stage.move_abs(axis, pos)

    def move_rel(self, axis: str, shift: int):
        print(f"[APP] move_rel {axis} by {shift} nm")
        self.stage.move_rel(axis, shift)

    def get_pos(self, axis: str) -> int:
        pos = self.stage.get_pos(axis)
        print(f"[APP] pos({axis}) = {pos} nm")
        return pos

    def scan_grid(self, axis_x: str, axis_y: str, x_positions: Iterable[int], y_positions: Iterable[int]) -> Iterable[Tuple[int,int,int,int]]:
        """
        Example high-level routine: scan over a grid of (x_positions, y_positions).
        Yields tuples: (x_pos, y_pos, x_readback, y_readback)
        """
        for xp in x_positions:
            self.move_abs(axis_x, xp)
            for yp in y_positions:
                self.move_abs(axis_y, yp)
                xr = self.get_pos(axis_x)
                yr = self.get_pos(axis_y)
                yield xp, yp, xr, yr

    def close(self):
        self.stage.close()
