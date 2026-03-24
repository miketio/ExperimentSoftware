# hardware_control/setup_motor/filter_stage.py
"""
Filter Stage - Single-axis MCS controller for K-space filtering

FIXES:
- Calls SA_CalibrateSensor_S on init so absolute moves work
- Removed SA_SetPositionLimit_S (Python-level limits are sufficient,
  hardware limits require referencing first and caused error 129)
- Falls back gracefully if calibration not supported by sensor type
"""

import ctypes as ct
import time
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Dict
from datetime import datetime

import hardware_control.setup_motor.MCSControl_PythonWrapper as mcs


class FilterStage:
    """
    Single-axis MCS stage for K-space filtering.

    Features:
    - 1D sweep with configurable range and step
    - Image capture at each position
    - Metadata logging
    - Progress callbacks
    - Multi-step movement for large distances

    Usage:
        stage = FilterStage(locator="usb:id:1234")

        # Run sweep
        results = stage.run_sweep(
            start_nm=-15000000,
            end_nm=15000000,
            step_nm=1000,
            camera=camera,
            output_dir="data/sweep_001"
        )
    """

    # Software position limits — enforced in Python, no hardware command needed
    POSITION_LIMIT_MIN_NM = -15_000_000  # -15 mm
    POSITION_LIMIT_MAX_NM =  15_000_000  # +15 mm

    # Multi-step movement threshold
    LARGE_MOVE_THRESHOLD_NM = 1_000_000  # 1 mm
    LARGE_MOVE_STEP_NM      =   500_000  # 500 µm per step

    def __init__(
        self,
        locator: str,
        axis_channel: int = 0,
        options: str = "sync,reset"
    ):
        """
        Initialize filter stage.

        Args:
            locator:      MCS device locator (e.g., "usb:id:1234")
            axis_channel: Channel index for the filter axis (default: 0)
            options:      MCS open options (default: "sync,reset")
        """
        self.locator      = locator
        self.axis_channel = axis_channel
        self._closed      = False

        # MCS handle
        self.mcsHandle = ct.c_ulong()

        # Open system
        status = mcs.SA_OpenSystem(
            self.mcsHandle,
            locator.encode('utf-8'),
            options.encode('utf-8')
        )
        self._exit_if_error(status)

        # Query channel count
        num_channels = ct.c_ulong()
        status = mcs.SA_GetNumberOfChannels(self.mcsHandle, num_channels)
        self._exit_if_error(status)
        self.num_channels = num_channels.value

        print(f"[FilterStage] Initialized on {locator}")
        print(f"  Channels:     {self.num_channels}")
        print(f"  Axis channel: {axis_channel}")

        # Check sensor type (informational only)
        sensor_type = ct.c_ulong()
        status = mcs.SA_GetSensorType_S(
            self.mcsHandle,
            ct.c_ulong(axis_channel),
            sensor_type
        )
        if status == mcs.SA_OK:
            print(f"  Sensor type:  {sensor_type.value}")

        # Calibrate sensor so absolute moves work
        self._calibrate_sensor()

    # =========================================================================
    # Initialisation helpers
    # =========================================================================

    def _calibrate_sensor(self):
        """
        Run SA_CalibrateSensor_S so the controller knows the absolute position
        and can execute SA_GotoPositionAbsolute_S.

        Without this, the stage returns SA_INVALID_STATE_ERROR (129) on any
        absolute-position command.  The call is safe to ignore if the sensor
        type does not support calibration.
        """
        print("[FilterStage] Calibrating sensor (required for absolute moves)...")

        channel = ct.c_ulong(self.axis_channel)
        status  = mcs.SA_CalibrateSensor_S(self.mcsHandle, channel)

        if status == mcs.SA_OK:
            print("  ✅ Sensor calibrated — absolute positioning enabled")
        else:
            # Some sensor types (e.g. open-loop steppers) don't support
            # calibration.  That is fine; moves will still work in relative mode.
            print(f"  ⚠️  Sensor calibration not supported (code {status}) — "
                  f"falling back to relative moves if needed")

    # =========================================================================
    # Error handling
    # =========================================================================

    def _exit_if_error(self, status: int):
        """Raise RuntimeError if status != SA_OK."""
        if status == mcs.SA_OK:
            return

        try:
            err_buf = ct.create_string_buffer(256)
            mcs.SA_GetStatusInfo(ct.c_ulong(status), err_buf)
            msg = err_buf.value.decode('utf-8', errors='ignore')
        except Exception:
            msg = f"MCS error code {status}"

        raise RuntimeError(f"FilterStage error: {msg}")

    # =========================================================================
    # Basic Movement
    # =========================================================================

    def move_abs(self, pos_nm: int, hold_time_ms: int = 0, verify: bool = True):
        """
        Move to absolute position in nanometres.

        Args:
            pos_nm:       Target position (nm)
            hold_time_ms: Hold time after movement (ms)
            verify:       If True, verify final position is within 1 µm
        """
        # Python-level range check (no hardware limit command needed)
        if not (self.POSITION_LIMIT_MIN_NM <= pos_nm <= self.POSITION_LIMIT_MAX_NM):
            raise ValueError(
                f"Position {pos_nm} nm ({pos_nm/1e6:.3f} mm) outside limits "
                f"[{self.POSITION_LIMIT_MIN_NM/1e6:.3f}, "
                f"{self.POSITION_LIMIT_MAX_NM/1e6:.3f}] mm"
            )

        current_pos = self.get_position()
        distance    = abs(pos_nm - current_pos)

        print(f"[FilterStage] Moving from {current_pos/1000:.3f} to {pos_nm/1000:.3f} µm "
              f"(distance: {distance/1000:.3f} µm)")

        if distance > self.LARGE_MOVE_THRESHOLD_NM:
            print(f"  Large move ({distance/1e6:.3f} mm) — using multi-step approach")
            self._move_abs_multistep(current_pos, pos_nm, hold_time_ms)
        else:
            self._move_abs_direct(pos_nm, hold_time_ms)

        if verify:
            final_pos = self.get_position()
            error     = abs(final_pos - pos_nm)
            if error > 1000:  # 1 µm tolerance
                print(f"  ⚠️  Position error: {error} nm ({error/1000:.3f} µm) — "
                      f"target {pos_nm} nm, actual {final_pos} nm")
            else:
                print(f"  ✅ Position verified: {final_pos} nm (error: {error} nm)")

    def _move_abs_direct(self, pos_nm: int, hold_time_ms: int = 0):
        """Direct absolute move (internal)."""
        channel  = ct.c_ulong(self.axis_channel)
        position = ct.c_long(int(pos_nm))
        hold     = ct.c_ulong(hold_time_ms)

        status = mcs.SA_GotoPositionAbsolute_S(
            self.mcsHandle, channel, position, hold
        )
        self._exit_if_error(status)
        self._wait_for_stop()

    def _move_abs_multistep(self, start_nm: int, target_nm: int, hold_time_ms: int = 0):
        """Move in multiple steps for large distances."""
        distance  = abs(target_nm - start_nm)
        num_steps = int(np.ceil(distance / self.LARGE_MOVE_STEP_NM))

        print(f"  Multi-step move: {num_steps} steps of "
              f"~{self.LARGE_MOVE_STEP_NM/1000:.1f} µm")

        positions = np.linspace(start_nm, target_nm, num_steps + 1)[1:]

        for i, pos in enumerate(positions):
            pos_int = int(pos)
            print(f"    Step {i+1}/{num_steps}: {pos_int/1000:.3f} µm")
            self._move_abs_direct(
                pos_int,
                hold_time_ms if i == len(positions) - 1 else 0
            )
            if i < len(positions) - 1:
                time.sleep(0.1)

    def move_rel(self, shift_nm: int, hold_time_ms: int = 0):
        """
        Move relative to current position.

        Args:
            shift_nm:     Distance to move in nanometres (can be negative)
            hold_time_ms: Hold time after movement (ms)
        """
        current_pos = self.get_position()
        target_pos  = current_pos + shift_nm
        self.move_abs(target_pos, hold_time_ms)

    def _wait_for_stop(self, timeout_s: float = 30.0):
        """Wait for stage to finish moving."""
        channel    = ct.c_ulong(self.axis_channel)
        status_val = ct.c_ulong()
        start_time = time.time()

        while True:
            status = mcs.SA_GetStatus_S(self.mcsHandle, channel, status_val)
            self._exit_if_error(status)

            if status_val.value in (mcs.SA_STOPPED_STATUS, mcs.SA_TARGET_STATUS):
                break

            if time.time() - start_time > timeout_s:
                raise RuntimeError(f"Move timeout after {timeout_s}s")

            time.sleep(0.05)

        time.sleep(0.2)  # final settle

    # =========================================================================
    # Position query
    # =========================================================================

    def get_position(self) -> int:
        """Return current position in nanometres."""
        channel  = ct.c_ulong(self.axis_channel)
        position = ct.c_long()

        status = mcs.SA_GetPosition_S(self.mcsHandle, channel, position)
        self._exit_if_error(status)

        return int(position.value)

    # =========================================================================
    # Sweep Operation
    # =========================================================================

    def run_sweep(
        self,
        start_nm: int,
        end_nm: int,
        step_nm: int,
        camera,
        output_dir: str,
        settle_time_s: float = 0.5,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict:
        """
        Run 1D sweep and capture images.

        Args:
            start_nm:          Start position (nm)
            end_nm:            End position (nm)
            step_nm:           Step size (nm)
            camera:            Camera instance with acquire_single_image()
            output_dir:        Directory to save images and metadata
            settle_time_s:     Wait time after each move (s)
            progress_callback: Optional function(current, total)

        Returns:
            dict with sweep results
        """
        print(f"[FilterStage] Starting sweep:")
        print(f"  Range: {start_nm} to {end_nm} nm "
              f"({(end_nm - start_nm)/1e6:.3f} mm)")
        print(f"  Step:  {step_nm} nm ({step_nm/1000:.3f} µm)")

        if not (self.POSITION_LIMIT_MIN_NM <= start_nm <= self.POSITION_LIMIT_MAX_NM):
            raise ValueError(f"Start position {start_nm} nm outside limits")
        if not (self.POSITION_LIMIT_MIN_NM <= end_nm <= self.POSITION_LIMIT_MAX_NM):
            raise ValueError(f"End position {end_nm} nm outside limits")

        positions     = list(range(start_nm, end_nm + step_nm, step_nm))
        num_positions = len(positions)
        print(f"  Total positions: {num_positions}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        start_time       = datetime.now()
        image_files      = []
        actual_positions = []

        try:
            for idx, target_pos in enumerate(positions):
                self.move_abs(target_pos, verify=False)
                time.sleep(settle_time_s)

                actual_pos = self.get_position()
                actual_positions.append(actual_pos)

                try:
                    image = camera.acquire_single_image()

                    image_filename = f"img_{idx:04d}_pos_{actual_pos}nm.tif"
                    image_path     = output_path / image_filename

                    import tifffile
                    tifffile.imwrite(str(image_path), image)
                    image_files.append(str(image_path))

                    print(f"  [{idx+1}/{num_positions}] "
                          f"Pos={actual_pos}nm ({actual_pos/1000:.3f}µm) "
                          f"→ {image_filename}")

                except Exception as e:
                    print(f"  ⚠️  Failed to capture at {actual_pos}nm: {e}")
                    image_files.append(None)

                if progress_callback:
                    progress_callback(idx + 1, num_positions)

        except KeyboardInterrupt:
            print("\n[FilterStage] ⚠️  Sweep interrupted by user")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        import json
        metadata = {
            'sweep_config': {
                'start_nm':            start_nm,
                'end_nm':              end_nm,
                'step_nm':             step_nm,
                'requested_positions': num_positions
            },
            'actual_data': {
                'target_positions_nm': positions,
                'actual_positions_nm': actual_positions,
                'image_files':         image_files
            },
            'timing': {
                'start_time':      start_time.isoformat(),
                'end_time':        end_time.isoformat(),
                'duration_seconds': duration,
                'settle_time_s':   settle_time_s
            },
            'hardware': {
                'mcs_locator':        self.locator,
                'axis_channel':       self.axis_channel,
                'position_limits_nm': {
                    'min': self.POSITION_LIMIT_MIN_NM,
                    'max': self.POSITION_LIMIT_MAX_NM
                }
            }
        }

        metadata_file = output_path / "sweep_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n[FilterStage] ✅ Sweep complete!")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Images saved: "
              f"{len([f for f in image_files if f is not None])}/{num_positions}")
        print(f"  Output: {output_dir}")

        return {
            'positions':     actual_positions,
            'image_files':   image_files,
            'metadata_file': str(metadata_file),
            'start_time':    start_time.isoformat(),
            'duration_s':    duration
        }

    # =========================================================================
    # Convenience
    # =========================================================================

    def home(self):
        """Move to zero position."""
        print("[FilterStage] Moving to home (0 nm)...")
        self.move_abs(0)

    def get_status(self) -> Dict:
        """Get stage status dict."""
        try:
            pos     = self.get_position()
            channel = ct.c_ulong(self.axis_channel)
            status  = ct.c_ulong()
            mcs.SA_GetStatus_S(self.mcsHandle, channel, status)

            status_names = {
                mcs.SA_STOPPED_STATUS:    'STOPPED',
                mcs.SA_STEPPING_STATUS:   'STEPPING',
                mcs.SA_SCANNING_STATUS:   'SCANNING',
                mcs.SA_HOLDING_STATUS:    'HOLDING',
                mcs.SA_TARGET_STATUS:     'TARGET',
                mcs.SA_MOVE_DELAY_STATUS: 'MOVE_DELAY',
                mcs.SA_CALIBRATING_STATUS:'CALIBRATING',
                mcs.SA_FINDING_REF_STATUS:'FINDING_REF'
            }

            return {
                'connected':   not self._closed,
                'position_nm': pos,
                'position_um': pos / 1000.0,
                'status':      status_names.get(status.value, f'UNKNOWN({status.value})'),
                'locator':     self.locator,
                'channel':     self.axis_channel,
            }
        except Exception as e:
            return {'connected': False, 'error': str(e)}

    def print_status(self):
        """Print stage status to console."""
        s = self.get_status()
        print("\n" + "=" * 60)
        print("FilterStage Status")
        print("=" * 60)
        print(f"Locator:   {s.get('locator', 'N/A')}")
        print(f"Channel:   {s.get('channel', 'N/A')}")
        print(f"Connected: {s.get('connected', False)}")
        if s.get('connected'):
            print(f"Position:  {s['position_nm']} nm ({s['position_um']:.3f} µm)")
            print(f"Status:    {s.get('status', 'UNKNOWN')}")
        else:
            print(f"Error:     {s.get('error', 'Unknown')}")
        print("=" * 60 + "\n")

    # =========================================================================
    # Cleanup
    # =========================================================================

    def close(self):
        """Close MCS connection."""
        if self._closed:
            return
        try:
            mcs.SA_CloseSystem(self.mcsHandle)
        finally:
            self._closed = True
            print("[FilterStage] Closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()