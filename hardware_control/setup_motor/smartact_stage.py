# smartact_stage.py
import ctypes as ct
import time
from typing import Dict, Optional

from hardware_control.setup_motor.xyz_stage_base import XYZStageBase

# Import your MCS wrapper here; name must match your environment
import hardware_control.setup_motor.MCSControl_PythonWrapper as mcs


class SmarActXYZStage(XYZStageBase):
    """
    Concrete implementation for the SmarAct MCS controller with built-in precision movement.
    
    Features:
      - 3-stage precision positioning (coarse → fine → correction)
      - Configurable axis mapping (default: {'x': 0, 'y': 1, 'z': 2})
      - Positions and shifts in nanometers (int)
      - Multi-MCS support via explicit locator parameter
      - Automatic error correction for high-accuracy positioning
      
    Movement Strategy:
      1. Coarse approach: 300µm steps until within 1mm of target
      2. Fine approach: Single move to target position
      3. Error correction: Iteratively correct until error < 100nm
      
    Usage:
        stage = SmarActXYZStage()
        stage.move_abs('x', 5_000_000)  # Automatically uses precision movement
        pos = stage.get_pos('x')
    """

    def __init__(
        self, 
        axis_map: Optional[Dict[str, int]] = None, 
        locator: Optional[str] = None,
        options: str = "sync",
        find_buffer_size: int = 256,
        # Precision movement parameters
        coarse_step_nm: int = 300_000,
        fine_threshold_nm: int = 1_000,
        correction_threshold_nm: int = 200,
        max_corrections: int = 3,
        position_tolerance_nm: int = 100,
        settle_time_s: float = 0.05,
        correction_settle_time_s: float = 0.02,
        verbose: bool = True
    ):
        """
        Initialize SmarAct XYZ stage with precision movement.
        
        Args:
            axis_map: Dict mapping axis names to channel indices 
                     (default: {'x': 0, 'y': 1, 'z': 2})
            locator: Optional MCS device locator string (e.g., "usb:id:1234")
                    If None, auto-discovers first available device
            options: MCS open options (default: "sync")
            find_buffer_size: Buffer size for device discovery (default: 256)
            
            Precision movement parameters:
            coarse_step_nm: Step size for coarse approach (default: 300µm)
            fine_threshold_nm: Distance to switch to fine approach (default: 1mm)
            correction_threshold_nm: Error threshold for correction (default: 5µm)
            max_corrections: Maximum correction iterations (default: 3)
            position_tolerance_nm: Success tolerance (default: 100nm)
            settle_time_s: Settle time after moves (default: 50ms)
            correction_settle_time_s: Settle time after corrections (default: 20ms)
            verbose: Print detailed movement information (default: False)
        """
        # handle & initialization
        self.mcsHandle = ct.c_ulong()
        self._closed = False

        # axis name -> channel index mapping (default from your original)
        self.axis_dict = axis_map or {'x': 0, 'y': 1, 'z': 2}

        # Precision movement configuration
        self.coarse_step_nm = coarse_step_nm
        self.fine_threshold_nm = fine_threshold_nm
        self.correction_threshold_nm = correction_threshold_nm
        self.max_corrections = max_corrections
        self.position_tolerance_nm = position_tolerance_nm
        self.settle_time_s = settle_time_s
        self.correction_settle_time_s = correction_settle_time_s
        self.verbose = verbose

        # If locator provided, use it directly
        if locator is not None:
            print(f"[SmarActXYZStage] Opening specified device: {locator}")
            self._open_specific_device(locator, options)
        else:
            # Auto-discover first device (original behavior)
            print("[SmarActXYZStage] Auto-discovering MCS device...")
            self._auto_discover_and_open(options, find_buffer_size)
        
        # Query device info
        self._query_device_info()
        
        # Print precision config
        if self.verbose:
            print(f"[SmarActXYZStage] Precision movement enabled:")
            print(f"  Coarse steps: {self.coarse_step_nm/1000:.1f} µm")
            print(f"  Fine threshold: {self.fine_threshold_nm/1000:.1f} µm")
            print(f"  Correction threshold: {self.correction_threshold_nm/1000:.3f} µm")
            print(f"  Max corrections: {self.max_corrections}")
            print(f"  Target tolerance: {self.position_tolerance_nm} nm")

    def _open_specific_device(self, locator: str, options: str):
        """Open a specific MCS device by locator."""
        locator_bytes = locator.encode('utf-8')
        options_bytes = options.encode('utf-8')
        
        status = mcs.SA_OpenSystem(self.mcsHandle, locator_bytes, options_bytes)
        self._exit_if_error(status)
        
        print(f"[SmarActXYZStage] ✅ Opened device: {locator}")

    def _auto_discover_and_open(self, options: str, find_buffer_size: int):
        """Auto-discover and open first available MCS device."""
        buffer_size = max(256, find_buffer_size)
        outBuffer = ct.create_string_buffer(buffer_size)
        ioBufferSize = ct.c_ulong(buffer_size)

        # Try to find systems
        status = mcs.SA_FindSystems('', outBuffer, ioBufferSize)
        self._exit_if_error(status)

        # decode the returned string (value up to first null byte)
        found = outBuffer.value.decode('utf-8', errors='ignore')
        if not found:
            raise RuntimeError("No MCS systems found by SA_FindSystems().")
        
        # Handle multiple devices (comma or newline separated)
        found_cleaned = found.replace('\n', ',').replace('\r', ',')
        locators = [loc.strip() for loc in found_cleaned.split(',') if loc.strip()]
        
        if len(locators) > 1:
            print(f"[SmarActXYZStage] ⚠️  Multiple MCS devices found ({len(locators)})")
            print(f"[SmarActXYZStage] Using first device: {locators[0]}")
            print(f"[SmarActXYZStage] Hint: Use MultiMCSManager for multi-device control")
        
        first_locator = locators[0]
        print(f"[SmarActXYZStage] MCS address: {first_locator}")

        # open system in sync mode
        locator_bytes = first_locator.encode('utf-8')
        options_bytes = options.encode('utf-8')
        
        status = mcs.SA_OpenSystem(self.mcsHandle, locator_bytes, options_bytes)
        self._exit_if_error(status)

    def _query_device_info(self):
        """Query and print device information."""
        # get number of channels
        num_channels = ct.c_ulong()
        status = mcs.SA_GetNumberOfChannels(self.mcsHandle, num_channels)
        self._exit_if_error(status)
        print(f"[SmarActXYZStage] Number of Channels: {num_channels.value}")
        
        # Validate that we have 3 channels for XYZ stage
        if num_channels.value < 3:
            print(f"[SmarActXYZStage] ⚠️ WARNING: Expected 3 channels, found {num_channels.value}")
            print(f"[SmarActXYZStage] This may not be a valid XYZ stage!")

        # check sensor type on channel 0
        sensorType = ct.c_ulong()
        status = mcs.SA_GetSensorType_S(self.mcsHandle, ct.c_ulong(0), sensorType)
        self._exit_if_error(status)
        print(f"[SmarActXYZStage] Sensor type for channel 0: {sensorType.value}")

        # check physical position known for configured channels
        for axis_name, ch in self.axis_dict.items():
            known = ct.c_uint()
            try:
                result = mcs.SA_GetPhysicalPositionKnown_S(
                    self.mcsHandle, 
                    ct.c_ulong(ch), 
                    known
                )
                if result == mcs.SA_OK:
                    status_str = "KNOWN" if known.value else "UNKNOWN"
                    print(f"[SmarActXYZStage] Position in channel {ch} ({axis_name}): {status_str}")
            except Exception:
                # non-fatal: some devices/channels may not be present
                pass

    def _exit_if_error(self, status: int) -> None:
        """Raise RuntimeError if status != SA_OK and include status text when possible."""
        if status == mcs.SA_OK:
            return
        # attempt to get text of status
        try:
            err_buf = ct.create_string_buffer(256)
            try:
                mcs.SA_GetStatusInfo(ct.c_ulong(status), err_buf)
                msg = err_buf.value.decode('utf-8', errors='ignore')
            except Exception:
                msg = f"MCS error code {status}"
        except Exception:
            msg = f"MCS error code {status}"
        raise RuntimeError(msg)

    # =========================================================================
    # PRECISION MOVEMENT - Private Methods
    # =========================================================================

    def _move_abs_direct(self, axis: str, pos: int) -> None:
        """Direct absolute move without precision features (internal use only)."""
        if axis not in self.axis_dict:
            raise ValueError(f"Unknown axis '{axis}'")
        channel = ct.c_ulong(self.axis_dict[axis])
        status = mcs.SA_GotoPositionAbsolute_S(
            self.mcsHandle, 
            channel, 
            ct.c_long(pos), 
            ct.c_ulong(0)
        )
        self._exit_if_error(status)
        time.sleep(0.2)

    def _coarse_approach(self, axis: str, start_nm: int, target_nm: int) -> int:
        """
        Stage 1: Coarse approach using large steps.
        Returns number of coarse steps taken.
        """
        if self.verbose:
            print(f"\n[Stage 1/3] Coarse Approach ({self.coarse_step_nm/1000:.0f} µm steps)")
        
        current_pos = start_nm
        direction = 1 if target_nm > start_nm else -1
        step_count = 0
        
        while True:
            remaining = abs(target_nm - current_pos)
            
            if self.verbose:
                print(f"  Step {step_count + 1}: Position {current_pos/1000:.3f} µm, "
                      f"Remaining {remaining/1000:.3f} µm")
            
            # Stop when close enough for fine approach
            if remaining <= self.fine_threshold_nm:
                if self.verbose:
                    print(f"  → Within fine threshold ({self.fine_threshold_nm/1000:.0f} µm)")
                break
            
            # Calculate next step (don't overshoot into fine range)
            step_size = min(self.coarse_step_nm, remaining - self.fine_threshold_nm)
            next_pos = current_pos + (direction * step_size)
            
            # Execute move
            self._move_abs_direct(axis, next_pos)
            time.sleep(self.settle_time_s)
            
            # Read actual position
            current_pos = self.get_pos(axis)
            step_count += 1
            
            # Safety check
            if direction > 0 and current_pos >= target_nm:
                break
            if direction < 0 and current_pos <= target_nm:
                break
        
        return step_count

    def _error_correction(self, axis: str, target_nm: int) -> tuple[int, int, int]:
        """
        Stage 3: Iterative error correction.
        Returns (corrections_made, final_position_nm, final_error_nm).
        """
        if self.verbose:
            print(f"\n[Stage 3/3] Error Correction")
        
        corrections = 0
        
        for i in range(self.max_corrections):
            current_pos = self.get_pos(axis)
            error = target_nm - current_pos
            abs_error = abs(error)
            
            if self.verbose:
                print(f"  Iteration {i+1}: Position {current_pos/1000:.3f} µm, "
                      f"Error {error} nm ({error/1000:.3f} µm)")
            
            # Check if within tolerance
            if abs_error <= self.position_tolerance_nm:
                if self.verbose:
                    print(f"  ✅ Within tolerance ({self.position_tolerance_nm} nm)")
                return corrections, current_pos, abs_error
            
            # Check if correction needed
            if abs_error <= self.correction_threshold_nm:
                if self.verbose:
                    print(f"  → Error below correction threshold ({self.correction_threshold_nm} nm)")
                return corrections, current_pos, abs_error
            
            # Apply correction
            if self.verbose:
                print(f"  → Applying correction: {error} nm")
            
            corrected_target = current_pos + error
            self._move_abs_direct(axis, corrected_target)
            time.sleep(self.correction_settle_time_s)
            corrections += 1
        
        # Final readback
        final_pos = self.get_pos(axis)
        final_error = abs(target_nm - final_pos)
        
        if self.verbose:
            print(f"  ⚠️ Max corrections ({self.max_corrections}) reached")
        
        return corrections, final_pos, final_error

    # =========================================================================
    # PUBLIC API - With Built-in Precision Movement
    # =========================================================================

    def move_abs(self, axis: str, pos: int) -> None:
        """
        Absolute move with built-in 3-stage precision positioning.
        
        This method automatically uses:
        1. Coarse approach (300µm steps) for distances > 1mm
        2. Fine approach (single move to target)
        3. Error correction (iterative correction until < 100nm error)
        
        Args:
            axis: Axis name ('x', 'y', or 'z')
            pos: Target position in nanometers
            
        The precision behavior is configured at initialization and can be
        adjusted via set_precision_config().
        """
        if axis not in self.axis_dict:
            raise ValueError(f"Unknown axis '{axis}'")
        
        # Get starting position
        initial_pos = self.get_pos(axis)
        distance = abs(pos - initial_pos)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[Precision Move] Axis {axis.upper()}: {initial_pos/1000:.3f} → {pos/1000:.3f} µm")
            print(f"                 Distance: {distance/1000:.3f} µm ({distance/1e6:.3f} mm)")
            print(f"{'='*70}")
        
        # Stage 1: Coarse approach (if needed)
        coarse_steps = 0
        if distance > self.fine_threshold_nm:
            coarse_steps = self._coarse_approach(axis, initial_pos, pos)
        
        # Stage 2: Fine approach
        if self.verbose:
            current_pos = self.get_pos(axis)
            remaining = abs(pos - current_pos)
            print(f"\n[Stage 2/3] Fine Approach")
            print(f"  Current: {current_pos/1000:.3f} µm")
            print(f"  Target:  {pos/1000:.3f} µm")
            print(f"  Remaining: {remaining/1000:.3f} µm")
        
        self._move_abs_direct(axis, pos)
        time.sleep(self.settle_time_s)
        
        # Stage 3: Error correction
        corrections, final_pos, final_error = self._error_correction(axis, pos)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[Precision Move Complete]")
            print(f"  Initial position: {initial_pos/1000:.3f} µm")
            print(f"  Target position:  {pos/1000:.3f} µm")
            print(f"  Final position:   {final_pos/1000:.3f} µm")
            print(f"  Final error:      {final_error} nm ({final_error/1000:.3f} µm)")
            print(f"  Coarse steps:     {coarse_steps}")
            print(f"  Corrections:      {corrections}")
            success = final_error <= self.position_tolerance_nm
            print(f"  Status:           {'✅ SUCCESS' if success else '⚠️ TOLERANCE EXCEEDED'}")
            print(f"{'='*70}\n")

    def move_rel(self, axis: str, shift: int, sleep_time: float = 0.01) -> None:
        """
        Relative move using precision positioning.
        
        Args:
            axis: Axis name ('x', 'y', or 'z')
            shift: Relative shift in nanometers
            sleep_time: Legacy parameter (ignored, uses settle_time_s instead)
        """
        if axis not in self.axis_dict:
            raise ValueError(f"Unknown axis '{axis}'")
        
        # Get current position
        current_pos = self.get_pos(axis)
        
        # Calculate target
        target_pos = current_pos + shift
        
        # Use precision absolute move
        self.move_abs(axis, target_pos)

    def get_pos(self, axis: str) -> int:
        """Readback position (nm)"""
        if axis not in self.axis_dict:
            raise ValueError(f"Unknown axis '{axis}'")
        channel = ct.c_ulong(self.axis_dict[axis])
        position = ct.c_long()
        status = mcs.SA_GetPosition_S(self.mcsHandle, channel, position)
        self._exit_if_error(status)
        return int(position.value)

    def get_pos_all(self) -> Dict[str, int]:
        """Get positions of all configured axes."""
        return {axis: self.get_pos(axis) for axis in self.axis_dict.keys()}

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    def set_precision_config(
        self,
        coarse_step_nm: Optional[int] = None,
        fine_threshold_nm: Optional[int] = None,
        correction_threshold_nm: Optional[int] = None,
        max_corrections: Optional[int] = None,
        position_tolerance_nm: Optional[int] = None,
        settle_time_s: Optional[float] = None,
        correction_settle_time_s: Optional[float] = None,
        verbose: Optional[bool] = None
    ):
        """
        Update precision movement configuration.
        
        Args:
            coarse_step_nm: Step size for coarse approach (µm)
            fine_threshold_nm: Switch to fine at this distance (µm)
            correction_threshold_nm: Correct if error exceeds this (nm)
            max_corrections: Max correction iterations
            position_tolerance_nm: Success tolerance (nm)
            settle_time_s: Settle time after moves (seconds)
            correction_settle_time_s: Settle time after corrections (seconds)
            verbose: Enable detailed output
            
        Example presets:
            # Ultra-precision (slow, <50nm accuracy)
            stage.set_precision_config(
                coarse_step_nm=100_000,
                correction_threshold_nm=1000,
                max_corrections=5,
                position_tolerance_nm=50
            )
            
            # Fast mode (~1µm accuracy)
            stage.set_precision_config(
                coarse_step_nm=500_000,
                correction_threshold_nm=10000,
                max_corrections=1,
                position_tolerance_nm=500
            )
            
            # Coarse only (fastest, ~10µm accuracy)
            stage.set_precision_config(
                coarse_step_nm=1_000_000,
                fine_threshold_nm=1_000_000,
                correction_threshold_nm=1_000_000,
                max_corrections=0
            )
        """
        if coarse_step_nm is not None:
            self.coarse_step_nm = coarse_step_nm
        if fine_threshold_nm is not None:
            self.fine_threshold_nm = fine_threshold_nm
        if correction_threshold_nm is not None:
            self.correction_threshold_nm = correction_threshold_nm
        if max_corrections is not None:
            self.max_corrections = max_corrections
        if position_tolerance_nm is not None:
            self.position_tolerance_nm = position_tolerance_nm
        if settle_time_s is not None:
            self.settle_time_s = settle_time_s
        if correction_settle_time_s is not None:
            self.correction_settle_time_s = correction_settle_time_s
        if verbose is not None:
            self.verbose = verbose
        
        if verbose or self.verbose:
            print(f"[SmarActXYZStage] Precision config updated:")
            print(f"  Coarse steps: {self.coarse_step_nm/1000:.1f} µm")
            print(f"  Fine threshold: {self.fine_threshold_nm/1000:.1f} µm")
            print(f"  Correction threshold: {self.correction_threshold_nm/1000:.3f} µm")
            print(f"  Max corrections: {self.max_corrections}")
            print(f"  Target tolerance: {self.position_tolerance_nm} nm")

    def enable_verbose(self, enable: bool = True):
        """Enable or disable verbose output for all moves."""
        self.verbose = enable

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        status = mcs.SA_CloseSystem(self.mcsHandle)
        self._exit_if_error(status)
        self._closed = True
        print("[SmarActXYZStage] Stage closed.")


if __name__ == "__main__":
    """
    Quick smoke test with precision movement demonstration.
    """
    stage = None
    try:
        print("Attempting to initialize SmarAct stage with precision movement...")
        
        # Initialize with verbose output to see precision movement in action
        stage = SmarActXYZStage(verbose=True)
        
        print("\n" + "="*70)
        print("Initial positions:")
        for ax in ("x", "y", "z"):
            try:
                pos = stage.get_pos(ax)
                print(f"  Axis {ax}: {pos} nm ({pos/1000:.3f} µm)")
            except Exception as e:
                print(f"  Could not read axis {ax}: {e}")

        print("\n" + "="*70)
        print("Testing precision movement on X axis...")
        print("="*70)
        
        # Test 1: Small move (no coarse steps needed)
        print("\nTest 1: Small move (500µm)")
        stage.move_abs('x', 500_000)
        
        # Test 2: Large move (will use coarse steps)
        print("\nTest 2: Large move (5mm)")
        stage.move_abs('x', 5_000_000)
        
        # Test 3: Return to zero
        print("\nTest 3: Return to zero")
        stage.move_abs('x', 0)
        
        print("\n" + "="*70)
        print("Testing different precision configs...")
        print("="*70)
        
        # Test fast mode
        print("\nSwitching to FAST mode...")
        stage.set_precision_config(
            coarse_step_nm=500_000,
            correction_threshold_nm=10000,
            max_corrections=1,
            verbose=True
        )
        stage.move_abs('y', 10_000_000)  # 10mm
        
        # Test ultra-precision mode
        print("\nSwitching to ULTRA-PRECISION mode...")
        stage.set_precision_config(
            coarse_step_nm=100_000,
            correction_threshold_nm=1000,
            max_corrections=5,
            position_tolerance_nm=50,
            verbose=True
        )
        stage.move_abs('y', 5_000_000)  # 5mm
        
        print("\n" + "="*70)
        print("✅ All tests complete!")
        print("="*70)

    except Exception as e:
        print("Failed to initialize or communicate with SmarAct stage:")
        print(" ", repr(e))
        import traceback
        traceback.print_exc()
    finally:
        if stage is not None:
            try:
                stage.close()
            except Exception as e:
                print("Error while closing stage:", e)
        print("Done.")