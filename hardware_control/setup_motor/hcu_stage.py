# hardware_control/setup_motor/hcu_stage.py
"""
SmarAct HCU-3C Stage Driver  —  via pylablib SCU interface

Uses pylablib.devices.SmarAct.SCUStage which wraps the SCU SDK.
No custom ctypes wrapper required.

Install dependency:
    pip install pylablib

UNIT CONTRACT
─────────────
  XYZStageBase (and the rest of your codebase) operates in NANOMETERS.
  pylablib SCUStage.get_position / move_to / move_by operate in METERS.
  All conversion happens here, invisibly to the caller.

    XYZStageBase boundary  →  nm
    pylablib internal       →  m
    Conversion:   m = nm / 1e9
                  nm = m * 1e9
"""

import time
from typing import Dict, Optional

from pylablib.devices import SmarAct

from hardware_control.setup_motor.xyz_stage_base import XYZStageBase


class HCU3CStage(XYZStageBase):
    """
    SmarAct HCU-3C stage (SCU controller) using pylablib.

    All public methods accept/return positions in NANOMETERS,
    matching the existing XYZStageBase contract used everywhere
    in the codebase.

    Usage:
        stage = HCU3CStage()                     # auto-picks first device
        stage = HCU3CStage(device_id=1679365169) # explicit device_id
        stage.move_abs('y', 5_000_000)           # 5 mm in nm
        pos = stage.get_pos('y')                 # returns nm
        stage.close()
    """

    def __init__(
        self,
        device_id: Optional[int] = None,
        axis_map:  Optional[Dict[str, int]] = None,
        verbose:   bool = True,
    ):
        """
        Args:
            device_id:  Integer device ID from list_scu_devices().
                        Auto-discovers if None.
            axis_map:   Maps axis names to channel indices.
                        Default: {'x': 0, 'y': 1, 'z': 2}
            verbose:    Print progress messages.
        """
        self._closed  = False
        self.verbose  = verbose
        self.axis_map = axis_map or {"x": 0, "y": 1, "z": 2}

        # ── Discover / validate device ────────────────────────────────
        devices = SmarAct.list_scu_devices()

        if not devices:
            raise RuntimeError(
                "No SmarAct SCU devices found.\n"
                "Check USB connection and that the SCU driver is installed."
            )

        if device_id is None:
            chosen = devices[0]
            if len(devices) > 1:
                print(f"[HCU3CStage] ⚠️  {len(devices)} SCU devices found — "
                      f"using first: device_id={chosen.device_id}")
                print("             Pass device_id= to choose explicitly.")
        else:
            matches = [d for d in devices if d.device_id == device_id]
            if not matches:
                ids = [d.device_id for d in devices]
                raise RuntimeError(
                    f"device_id={device_id} not found. Available: {ids}"
                )
            chosen = matches[0]

        self.device_id = chosen.device_id

        if verbose:
            print(f"[HCU3CStage] Connecting to device_id={chosen.device_id}  "
                  f"fw={chosen.firmware_version}  dll={chosen.dll_version}")

        # ── Open connection ───────────────────────────────────────────
        self._stage = SmarAct.SCUStage(chosen.device_id)

        if verbose:
            self._print_info()

    # =================================================================
    # XYZStageBase interface  (units: NANOMETERS)
    # =================================================================

    def move_abs(self, axis: str, pos_nm: int) -> None:
        """
        Move axis to absolute position.

        Args:
            axis:   'x', 'y', or 'z'
            pos_nm: Target position in nanometers
        """
        ch    = self._ch(axis)
        pos_m = int(pos_nm) / 1e9          # nm → m for pylablib

        if self.verbose:
            current_nm = self.get_pos(axis)
            delta_um   = abs(pos_nm - current_nm) / 1e3
            print(f"[HCU3CStage] move_abs {axis.upper()}: "
                  f"{current_nm/1e6:.3f} → {pos_nm/1e6:.3f} mm  "
                  f"(Δ={delta_um:.2f} µm)")

        self._stage.move_to(ch, pos_m)
        self._stage.wait_move(ch)

    def move_rel(self, axis: str, shift_nm: int, sleep_time: float = 0.0) -> None:
        """
        Move axis by a relative amount.

        Args:
            axis:       'x', 'y', or 'z'
            shift_nm:   Displacement in nanometers (can be negative)
            sleep_time: Unused — kept for API compatibility
        """
        ch      = self._ch(axis)
        shift_m = int(shift_nm) / 1e9      # nm → m

        if self.verbose:
            print(f"[HCU3CStage] move_rel {axis.upper()}: "
                  f"Δ={shift_nm/1e6:.4f} mm ({shift_nm:+,} nm)")

        self._stage.move_by(ch, shift_m)
        self._stage.wait_move(ch)

    def get_pos(self, axis: str) -> int:
        """
        Read back current position.

        Returns:
            Position in nanometers (int).
        """
        ch    = self._ch(axis)
        pos_m = self._stage.get_position(ch)   # pylablib → meters
        return int(round(pos_m * 1e9))          # m → nm

    def close(self) -> None:
        """Close the SCU connection."""
        if self._closed:
            return
        try:
            self._stage.close()
        finally:
            self._closed = True
            if self.verbose:
                print("[HCU3CStage] Closed.")

    # =================================================================
    # Convenience extras
    # =================================================================

    def get_pos_all(self) -> Dict[str, int]:
        """Return {axis: position_nm} for all configured axes."""
        return {ax: self.get_pos(ax) for ax in self.axis_map}

    def is_moving(self, axis: str) -> bool:
        """True while the axis is actively executing a move."""
        return self._stage.is_moving(self._ch(axis))

    def stop(self, axis: Optional[str] = None):
        """Stop one axis, or all configured axes if axis is None."""
        if axis is not None:
            self._stage.stop(self._ch(axis))
        else:
            for ch in self.axis_map.values():
                self._stage.stop(ch)

    def home(self, axis: str):
        """
        Find the home/reference mark for one axis (blocking).

        Args:
            axis: 'x', 'y', or 'z'
        """
        ch = self._ch(axis)
        print(f"[HCU3CStage] Homing {axis.upper()} (ch{ch}) …")
        self._stage.home(ch)
        self._stage.wait_move(ch, timeout=120.)
        print(f"[HCU3CStage] Homing {axis.upper()} done  "
              f"→ {self.get_pos(axis):,} nm")

    def print_status(self):
        """Print a human-readable status summary."""
        print("\n" + "=" * 60)
        print("HCU3CStage Status (pylablib SCU)")
        print("=" * 60)
        print(f"  device_id : {self.device_id}")
        print(f"  closed    : {self._closed}")
        for ax, ch in self.axis_map.items():
            try:
                pos_nm = self.get_pos(ax)
                moving = self._stage.is_moving(ch)
                print(f"  {ax.upper()} (ch{ch}): {pos_nm:>14,} nm  "
                      f"({pos_nm/1e6:>9.3f} mm)  "
                      f"{'MOVING' if moving else 'stopped'}")
            except Exception as e:
                print(f"  {ax.upper()} (ch{ch}): ERROR — {e}")
        print("=" * 60 + "\n")

    # =================================================================
    # Internal helpers
    # =================================================================

    def _ch(self, axis: str) -> int:
        ax = axis.lower()
        if ax not in self.axis_map:
            raise ValueError(
                f"Unknown axis '{axis}'. "
                f"Valid: {list(self.axis_map.keys())}"
            )
        return self.axis_map[ax]

    def _print_info(self):
        try:
            devices = SmarAct.list_scu_devices()
            info    = next((d for d in devices
                            if d.device_id == self.device_id), None)
            if info:
                print(f"[HCU3CStage] Firmware : {info.firmware_version}")
                print(f"[HCU3CStage] DLL      : {info.dll_version}")
        except Exception:
            pass
        print(f"[HCU3CStage] Axis map : {self.axis_map}")