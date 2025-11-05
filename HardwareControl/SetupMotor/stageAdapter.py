"""
stageAdapter.py - Micrometer interface for nanometer stage hardware.

Provides a clean µm-based API while hardware operates in nm.
All alignment/vision code works in µm and uses this adapter.
"""

class StageAdapterUM:
    """
    Adapter that wraps XYZStageBase and provides micrometer interface.
    
    Internal system uses micrometers (µm) for all calculations.
    Hardware (MockStage/real stage) operates in nanometers (nm).
    This adapter handles the conversion boundary.
    """
    
    def __init__(self, stage_nm):
        """
        Args:
            stage_nm: XYZStageBase instance (operates in nanometers)
        """
        self.stage_nm = stage_nm
        self._conversion = 1000.0  # 1 µm = 1000 nm
    
    def move_abs(self, axis: str, pos_um: float) -> None:
        """Move to absolute position in micrometers."""
        pos_nm = int(round(pos_um * self._conversion))
        self.stage_nm.move_abs(axis, pos_nm)
    
    def move_rel(self, axis: str, shift_um: float, sleep_time: float = 0.01) -> None:
        """Move relative in micrometers."""
        shift_nm = int(round(shift_um * self._conversion))
        self.stage_nm.move_rel(axis, shift_nm, sleep_time)
    
    def get_pos(self, axis: str) -> float:
        """Get position in micrometers."""
        pos_nm = self.stage_nm.get_pos(axis)
        return pos_nm / self._conversion
    
    def get_pos_all(self) -> dict:
        """Get all positions in micrometers."""
        pos_nm = self.stage_nm.get_pos_all()
        return {k: v / self._conversion for k, v in pos_nm.items()}
    
    def set_pos_all(self, x_um: float, y_um: float, z_um: float) -> None:
        """Set all axes in micrometers."""
        self.move_abs('x', x_um)
        self.move_abs('y', y_um)
        self.move_abs('z', z_um)
    
    def close(self) -> None:
        """Close underlying stage."""
        self.stage_nm.close()
    
    # Pass-through methods
    def set_camera_observer(self, camera):
        """Pass through to underlying stage."""
        self.stage_nm.set_camera_observer(camera)
    
    def get_status(self):
        """Get status (positions will be in µm)."""
        status = self.stage_nm.get_status()
        # Convert position dict to µm
        if 'position' in status:
            status['position'] = {k: v / self._conversion 
                                 for k, v in status['position'].items()}
        return status
if __name__ == "__main__":
    from HardwareControl.SetupMotor.mockStage_v2 import MockXYZStage
    stage_nm = MockXYZStage()
    stage_um = StageAdapterUM(stage_nm)
    stage_um.move_abs('x', 10.5)  # Move to 10.5 µm
    print("X Position (µm):", stage_um.get_pos('x'))
    print("Positions (nm):", stage_nm.get_pos('x')) # Directly from nm stage