# hardware_control/setup_motor/hcu3c_stage.py
"""
HCU-3C / SCU Stage — direct ctypes wrapper, mirrors smartact_stage.py.

Uses SCU3DControl.dll directly (via SCU3DControl_PythonWrapper.py) rather
than pylablib, giving full access to the C API including:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Closed-loop mode  (sensor present + referenced)                │
  │    move_abs → SA_MovePositionAbsolute_S  (real nm moves)        │
  │    move_rel → SA_MovePositionRelative_S  (real nm moves)        │
  │    get_pos  → SA_GetPosition_S           (real nm readback)     │
  ├─────────────────────────────────────────────────────────────────┤
  │  Open-loop mode  (no sensor / not yet referenced)               │
  │    move_abs → software-tracked step counter                     │
  │    move_rel → SA_MoveStep_S                                     │
  │    get_pos  → software-tracked estimate                         │
  └─────────────────────────────────────────────────────────────────┘

NOTE on DLL position units:
  SCU3DControl.dll uses 0.1 nm as its internal position unit, NOT 1 nm.
  All values sent to the DLL are multiplied by _DLL_UNITS_PER_NM (= 10),
  and all values read back are divided by _DLL_UNITS_PER_NM.
  The public API of this class always works in nanometers.

Requirements:
  SCU3DControl.dll on PATH or in working directory.

Usage:
    from hardware_control.setup_motor.hcu3c_stage import HCU3CStage

    # Closed-loop (sensor present, referenced):
    stage = HCU3CStage(device_idx=0)
    stage.reference('x')           # find reference mark → position known
    stage.move_abs('x', 5_000_000) # 5 mm
    print(stage.get_pos('x'))      # real nm readback

    # Open-loop (no sensor):
    stage = HCU3CStage(device_idx=0, nm_per_step=50.0)
    stage.move_rel('x', 10_000)    # ~10 µm (step-counted)
"""

import ctypes as ct
import time
from typing import Dict, Optional, Union

from hardware_control.setup_motor.xyz_stage_base import XYZStageBase
import hardware_control.setup_motor.SCU3DControl_PythonWrapper as scu


# ---------------------------------------------------------------------------
# SCU3DControl.dll uses 0.1 nm as its position unit.
# Multiply nm → DLL by 10; divide DLL → nm by 10.
# ---------------------------------------------------------------------------
_DLL_UNITS_PER_NM = 10


def _parse_version(v: int) -> str:
    return "{}.{}.{}.{}".format(
        (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF
    )


def list_hcu_devices() -> list:
    """
    Return info for every connected HCU/SCU device.

    Returns list of dicts:
        [{'device_idx': 0, 'device_id': 1679365169,
          'firmware': '1.5.0.9', 'dll': '5.5.0.217'}, ...]
    """
    status = scu.SA_InitDevices(scu.SA_SYNCHRONOUS_COMMUNICATION)
    if status == scu.SA_NO_DEVICES_FOUND_ERROR:
        return []
    if status != scu.SA_OK:
        raise RuntimeError(f"SA_InitDevices failed with code {status}")

    num = ct.c_uint()
    scu.SA_GetNumberOfDevices(num)

    dll_ver = ct.c_uint()
    scu.SA_GetDLLVersion(dll_ver)
    dll_str = _parse_version(dll_ver.value)

    devices = []
    for i in range(num.value):
        dev_id = ct.c_uint()
        fw_ver = ct.c_uint()
        scu.SA_GetDeviceID(ct.c_uint(i), dev_id)
        scu.SA_GetDeviceFirmwareVersion(ct.c_uint(i), fw_ver)
        devices.append({
            'device_idx': i,
            'device_id':  dev_id.value,
            'firmware':   _parse_version(fw_ver.value),
            'dll':        dll_str,
        })

    scu.SA_ReleaseDevices()
    return devices


class HCU3CStage(XYZStageBase):
    """
    SmarAct HCU-3C / SCU 3-axis stage controller.

    Mirrors the API of SmarActXYZStage, supporting both closed-loop
    (sensor present) and open-loop (step-counted) operation per axis.

    All public positions and shifts are in nanometers.
    The DLL-level unit conversion (_DLL_UNITS_PER_NM = 10) is handled
    internally and is invisible to callers.

    Parameters
    ----------
    device_idx : int
        SCU device index (0 = first device).
    axis_map : dict, optional
        {'x': channel, 'y': channel, 'z': channel}.
        Default: {'x': 0, 'y': 1, 'z': 2}.

    Closed-loop parameters
    ----------------------
    hold_time_ms : int
        Hold position after each closed-loop move (ms).  Default 0.
    settle_time_s : float
        Extra software delay after every move.  Default 0.05.
    move_timeout_s : float
        Maximum wait time per move.  Default 30.

    Open-loop parameters  (axes without sensors)
    ---------------------------------------------
    nm_per_step : float or dict
        Physical displacement per SA_MoveStep_S step.  Default 50 nm.
        Measure your stage to calibrate this.
    step_amplitude : int
        Drive voltage × 10  (1000 = 100 V).  Default 1000.
    step_frequency : int
        Step frequency in Hz.  Default 1000.
    """

    def __init__(
        self,
        device_idx: int = 0,
        axis_map: Optional[Dict[str, int]] = None,
        hold_time_ms: int = 0,
        settle_time_s: float = 0.05,
        move_timeout_s: float = 30.0,
        nm_per_step: Union[float, Dict[str, float]] = 50.0,
        step_amplitude: int = 1000,
        step_frequency: int = 1000,
        verbose: bool = True,
    ):
        self.device_idx     = ct.c_uint(device_idx)
        self.axis_map       = axis_map or {'x': 0, 'y': 1, 'z': 2}
        self.hold_time_ms   = hold_time_ms
        self.settle_time_s  = settle_time_s
        self.move_timeout_s = move_timeout_s
        self.step_amplitude = step_amplitude
        self.step_frequency = step_frequency
        self.verbose        = verbose
        self._closed        = False

        # Open-loop software step counter (used when no sensor)
        self._pos_steps: Dict[str, int] = {ax: 0 for ax in self.axis_map}

        # Per-axis nm_per_step calibration
        if isinstance(nm_per_step, dict):
            self._nm_per_step = {ax: float(nm_per_step.get(ax, 50.0)) for ax in self.axis_map}
        else:
            self._nm_per_step = {ax: float(nm_per_step) for ax in self.axis_map}

        # Initialize SCU library
        status = scu.SA_InitDevices(scu.SA_SYNCHRONOUS_COMMUNICATION)
        self._exit_if_error(status, "SA_InitDevices")

        # Validate device index
        num = ct.c_uint()
        scu.SA_GetNumberOfDevices(num)
        if device_idx >= num.value:
            scu.SA_ReleaseDevices()
            raise RuntimeError(
                f"Device index {device_idx} out of range — "
                f"only {num.value} device(s) found."
            )

        # Print device info
        dev_id  = ct.c_uint()
        fw_ver  = ct.c_uint()
        dll_ver = ct.c_uint()
        scu.SA_GetDeviceID(self.device_idx, dev_id)
        scu.SA_GetDeviceFirmwareVersion(self.device_idx, fw_ver)
        scu.SA_GetDLLVersion(dll_ver)
        print(f"[HCU3CStage] Connected: device_id={dev_id.value}"
              f"  fw={_parse_version(fw_ver.value)}"
              f"  dll={_parse_version(dll_ver.value)}"
              f"  (DLL unit = 0.1 nm, scale factor = {_DLL_UNITS_PER_NM})")

        # Detect sensor presence and position knowledge per axis
        self._has_sensor: Dict[str, bool] = {}
        self._pos_known:  Dict[str, bool] = {}
        for ax, ch in self.axis_map.items():
            ch_u    = ct.c_uint(ch)
            present = ct.c_uint()
            known   = ct.c_uint()
            scu.SA_GetSensorPresent_S(self.device_idx, ch_u, present)
            scu.SA_GetPhysicalPositionKnown_S(self.device_idx, ch_u, known)
            self._has_sensor[ax] = bool(present.value)
            self._pos_known[ax]  = bool(known.value)
            sensor_str = "sensor present" if self._has_sensor[ax] else "NO sensor (open-loop)"
            known_str  = "position KNOWN" if self._pos_known[ax] else "NOT referenced"
            print(f"  {ax.upper()} (ch {ch}): {sensor_str}, {known_str}")

        cl = [ax for ax in self.axis_map if self._has_sensor[ax]]
        ol = [ax for ax in self.axis_map if not self._has_sensor[ax]]
        if cl:
            print(f"[HCU3CStage] Closed-loop axes: {cl}")
        if ol:
            print(f"[HCU3CStage] Open-loop axes:   {ol}  (nm/step={self._nm_per_step})")

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _exit_if_error(self, status: int, context: str = "") -> None:
        if status == scu.SA_OK:
            return
        label = f" [{context}]" if context else ""
        raise RuntimeError(f"[HCU3CStage] SCU error code {status}{label}")

    # ------------------------------------------------------------------
    # Wait for motion to complete
    # ------------------------------------------------------------------

    def _wait_for_stop(self, axis: str, timeout_s: Optional[float] = None) -> None:
        """Poll channel status until STOPPED or HOLDING."""
        ch  = ct.c_uint(self.axis_map[axis])
        tmo = timeout_s if timeout_s is not None else self.move_timeout_s
        s   = ct.c_uint()
        t0  = time.time()
        while True:
            scu.SA_GetStatus_S(self.device_idx, ch, s)
            if s.value in (scu.SA_STOPPED_STATUS, scu.SA_HOLDING_STATUS):
                break
            if time.time() - t0 > tmo:
                raise RuntimeError(
                    f"[HCU3CStage] Timeout on {axis.upper()} after {tmo}s "
                    f"(status={s.value})"
                )
            time.sleep(0.02)

    # ------------------------------------------------------------------
    # XYZStageBase interface
    # ------------------------------------------------------------------

    def move_abs(self, axis: str, pos: int) -> None:
        """
        Move to absolute position in nanometers.

        Closed-loop: SA_MovePositionAbsolute_S.
        Open-loop:   delta computed from software counter, then move_rel.

        pos is in nm; internally multiplied by _DLL_UNITS_PER_NM before
        being passed to the DLL.
        """
        self._check_open()
        ch = ct.c_uint(self.axis_map[axis])

        if self._has_sensor[axis]:
            if self.verbose:
                try:
                    cur = self._get_pos_hw(axis)
                    print(f"[HCU3CStage] move_abs {axis.upper()}: {cur} nm → {pos} nm")
                except Exception:
                    print(f"[HCU3CStage] move_abs {axis.upper()} → {pos} nm")

            status = scu.SA_MovePositionAbsolute_S(
                self.device_idx, ch,
                int(pos) * _DLL_UNITS_PER_NM,   # nm → DLL units
                self.hold_time_ms
            )
            self._exit_if_error(status, "SA_MovePositionAbsolute_S")
            self._wait_for_stop(axis)
            self._pos_known[axis] = True

        else:
            delta_nm = pos - self.get_pos(axis)
            if delta_nm != 0:
                if self.verbose:
                    print(f"[HCU3CStage] move_abs {axis.upper()} (OL): "
                          f"{self.get_pos(axis)} nm → {pos} nm  ({delta_nm:+d} nm)")
                self.move_rel(axis, delta_nm)
                return   # move_rel handles settle

        if self.settle_time_s > 0:
            time.sleep(self.settle_time_s)

    def move_rel(self, axis: str, shift: int, sleep_time: float = 0.0) -> None:
        """
        Move by a relative shift in nanometers.

        Closed-loop: SA_MovePositionRelative_S.
        Open-loop:   SA_MoveStep_S + software counter update.

        shift is in nm; internally multiplied by _DLL_UNITS_PER_NM before
        being passed to the DLL.
        """
        self._check_open()
        if shift == 0:
            return
        ch = ct.c_uint(self.axis_map[axis])

        if self._has_sensor[axis]:
            if self.verbose:
                print(f"[HCU3CStage] move_rel {axis.upper()}: {shift:+d} nm")
            status = scu.SA_MovePositionRelative_S(
                self.device_idx, ch,
                int(shift) * _DLL_UNITS_PER_NM,  # nm → DLL units
                self.hold_time_ms
            )
            self._exit_if_error(status, "SA_MovePositionRelative_S")
            self._wait_for_stop(axis)

        else:
            steps = int(round(shift / self._nm_per_step[axis]))
            if steps == 0:
                if self.verbose:
                    print(f"[HCU3CStage] ⚠ {axis.upper()}: {shift} nm < 1 step — skipped.")
                return
            if self.verbose:
                print(f"[HCU3CStage] move_rel {axis.upper()} (OL): "
                      f"{shift:+d} nm  ({steps:+d} steps at {self.step_amplitude/10:.0f}V, "
                      f"{self.step_frequency}Hz)")
            status = scu.SA_MoveStep_S(
                self.device_idx, ch,
                steps, self.step_amplitude, self.step_frequency
            )
            self._exit_if_error(status, "SA_MoveStep_S")
            self._wait_for_stop(axis)
            self._pos_steps[axis] += steps

        if self.settle_time_s > 0:
            time.sleep(self.settle_time_s)

    def get_pos(self, axis: str) -> int:
        """
        Get current position in nanometers.

        Closed-loop: reads SA_GetPosition_S (real sensor value), divided
                     by _DLL_UNITS_PER_NM to convert DLL units → nm.
        Open-loop:   returns step counter × nm_per_step.
        """
        self._check_open()
        if self._has_sensor[axis]:
            return self._get_pos_hw(axis)
        return int(round(self._pos_steps[axis] * self._nm_per_step[axis]))

    def _get_pos_hw(self, axis: str) -> int:
        """Read hardware sensor position and return in nm."""
        pos = ct.c_int()
        status = scu.SA_GetPosition_S(
            self.device_idx, ct.c_uint(self.axis_map[axis]), pos
        )
        self._exit_if_error(status, "SA_GetPosition_S")
        return int(pos.value // _DLL_UNITS_PER_NM)  # DLL units → nm

    def close(self) -> None:
        """Release SCU library."""
        if self._closed:
            return
        try:
            scu.SA_ReleaseDevices()
        finally:
            self._closed = True
            print("[HCU3CStage] Closed.")

    # ------------------------------------------------------------------
    # Extended methods
    # ------------------------------------------------------------------

    def get_pos_all(self) -> Dict[str, int]:
        """Return {axis: pos_nm} for all axes."""
        return {ax: self.get_pos(ax) for ax in self.axis_map}

    def stop(self, axis: Optional[str] = None) -> None:
        """Emergency stop one or all axes."""
        self._check_open()
        for ax in ([axis] if axis else list(self.axis_map)):
            scu.SA_Stop_S(self.device_idx, ct.c_uint(self.axis_map[ax]))
        print(f"[HCU3CStage] Stopped: {axis or 'all'}")

    def reference(self, axis: str, auto_zero: bool = True, hold_time_ms: int = 0) -> None:
        """
        Find reference mark (requires sensor).

        auto_zero=True sets position to 0 at the reference mark.
        After this, get_pos() returns accurate absolute positions.
        """
        self._check_open()
        if not self._has_sensor[axis]:
            raise RuntimeError(f"Cannot reference {axis.upper()} — no sensor.")
        ch = ct.c_uint(self.axis_map[axis])
        print(f"[HCU3CStage] Referencing {axis.upper()}…")
        status = scu.SA_MoveToReference_S(
            self.device_idx, ch, hold_time_ms,
            scu.SA_AUTO_ZERO if auto_zero else scu.SA_NO_AUTO_ZERO
        )
        self._exit_if_error(status, "SA_MoveToReference_S")
        self._wait_for_stop(axis, timeout_s=60.0)
        self._pos_known[axis] = True
        print(f"[HCU3CStage] {axis.upper()} referenced → {self._get_pos_hw(axis)} nm")

    def calibrate(self, axis: str) -> None:
        """Run sensor calibration (requires sensor). Usually needed once after install."""
        self._check_open()
        if not self._has_sensor[axis]:
            raise RuntimeError(f"Cannot calibrate {axis.upper()} — no sensor.")
        ch = ct.c_uint(self.axis_map[axis])
        print(f"[HCU3CStage] Calibrating {axis.upper()}…")
        status = scu.SA_CalibrateSensor_S(self.device_idx, ch)
        self._exit_if_error(status, "SA_CalibrateSensor_S")
        self._wait_for_stop(axis, timeout_s=60.0)
        print(f"[HCU3CStage] {axis.upper()} calibration done.")

    def set_zero(self, axis: str) -> None:
        """Set current position as zero on a sensor axis (SA_SetZero_S)."""
        self._check_open()
        if not self._has_sensor[axis]:
            raise RuntimeError(f"Cannot set_zero on {axis.upper()} — no sensor.")
        scu.SA_SetZero_S(self.device_idx, ct.c_uint(self.axis_map[axis]))
        self._pos_known[axis] = True
        print(f"[HCU3CStage] {axis.upper()} zeroed at current position.")

    def zero_open_loop(self, axis: Optional[str] = None) -> None:
        """Reset open-loop software counter to 0 (no movement)."""
        for ax in ([axis] if axis else list(self.axis_map)):
            self._pos_steps[ax] = 0
        print(f"[HCU3CStage] Open-loop counter reset: {axis or 'all'}")

    def get_status(self, axis: str) -> str:
        """Return human-readable status string for an axis."""
        self._check_open()
        s = ct.c_uint()
        scu.SA_GetStatus_S(self.device_idx, ct.c_uint(self.axis_map[axis]), s)
        return {
            scu.SA_STOPPED_STATUS:             'STOPPED',
            scu.SA_SETTING_AMPLITUDE_STATUS:   'SETTING_AMPLITUDE',
            scu.SA_MOVING_STATUS:              'MOVING',
            scu.SA_TARGETING_STATUS:           'TARGETING',
            scu.SA_HOLDING_STATUS:             'HOLDING',
            scu.SA_CALIBRATING_STATUS:         'CALIBRATING',
            scu.SA_MOVING_TO_REFERENCE_STATUS: 'MOVING_TO_REFERENCE',
        }.get(s.value, f'UNKNOWN({s.value})')

    def print_status(self) -> None:
        """Print a full status table."""
        print("\n" + "=" * 65)
        print("HCU3CStage Status")
        print("=" * 65)
        print(f"Device idx:  {self.device_idx.value}  |  Connected: {not self._closed}")
        print()
        for ax, ch in self.axis_map.items():
            pos_nm = self.get_pos(ax)
            mode   = "closed-loop" if self._has_sensor[ax] else "open-loop"
            known  = "referenced" if self._pos_known[ax] else "NOT referenced"
            status = self.get_status(ax)
            print(f"  {ax.upper()} (ch {ch}): {pos_nm:>12,} nm  ({pos_nm/1000:.3f} µm)"
                  f"  [{mode}]  [{known}]  status={status}")
        print("=" * 65 + "\n")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("[HCU3CStage] Stage is already closed.")