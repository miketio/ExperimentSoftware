# hardware_control/setup_motor/filter_stage.py
"""
Filter Stage - Single-axis MCS controller for K-space filtering

This stage performs 1D sweeps to capture spectral data.
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
    
    Usage:
        stage = FilterStage(locator="usb:id:1234")
        
        # Run sweep
        results = stage.run_sweep(
            start_nm=0,
            end_nm=100000,
            step_nm=1000,
            camera=camera,
            output_dir="data/sweep_001"
        )
    """
    
    def __init__(
        self,
        locator: str,
        axis_channel: int = 0,
        options: str = "sync,reset"
    ):
        """
        Initialize filter stage.
        
        Args:
            locator: MCS device locator (e.g., "usb:id:1234")
            axis_channel: Channel index for the filter axis (default: 0)
            options: MCS open options (default: "sync,reset")
        """
        self.locator = locator
        self.axis_channel = axis_channel
        self._closed = False
        
        # MCS handle
        self.mcsHandle = ct.c_ulong()
        
        # Open system
        locator_bytes = locator.encode('utf-8')
        options_bytes = options.encode('utf-8')
        
        status = mcs.SA_OpenSystem(self.mcsHandle, locator_bytes, options_bytes)
        self._exit_if_error(status)
        
        # Query info
        num_channels = ct.c_ulong()
        status = mcs.SA_GetNumberOfChannels(self.mcsHandle, num_channels)
        self._exit_if_error(status)
        
        self.num_channels = num_channels.value
        
        print(f"[FilterStage] Initialized on {locator}")
        print(f"  Channels: {self.num_channels}")
        print(f"  Axis channel: {axis_channel}")
        
        # Check sensor
        sensor_type = ct.c_ulong()
        status = mcs.SA_GetSensorType_S(
            self.mcsHandle,
            ct.c_ulong(axis_channel),
            sensor_type
        )
        if status == mcs.SA_OK:
            print(f"  Sensor type: {sensor_type.value}")
    
    def _exit_if_error(self, status: int):
        """Raise RuntimeError if status != SA_OK."""
        if status == mcs.SA_OK:
            return
        
        try:
            err_buf = ct.create_string_buffer(256)
            mcs.SA_GetStatusInfo(ct.c_ulong(status), err_buf)
            msg = err_buf.value.decode('utf-8', errors='ignore')
        except:
            msg = f"MCS error code {status}"
        
        raise RuntimeError(f"FilterStage error: {msg}")
    
    # =========================================================================
    # Basic Movement
    # =========================================================================
    
    def move_abs(self, pos_nm: int, hold_time_ms: int = 0):
        """
        Move to absolute position.
        
        Args:
            pos_nm: Target position in nanometers
            hold_time_ms: Hold time after movement (milliseconds)
        """
        channel = ct.c_ulong(self.axis_channel)
        position = ct.c_long(int(pos_nm))
        hold = ct.c_ulong(hold_time_ms)
        
        status = mcs.SA_GotoPositionAbsolute_S(
            self.mcsHandle,
            channel,
            position,
            hold
        )
        self._exit_if_error(status)
        
        time.sleep(0.2)  # Settle time
    
    def move_rel(self, shift_nm: int, hold_time_ms: int = 0):
        """
        Move relative to current position.
        
        Args:
            shift_nm: Distance to move in nanometers
            hold_time_ms: Hold time after movement (milliseconds)
        """
        channel = ct.c_ulong(self.axis_channel)
        shift = ct.c_long(int(shift_nm))
        hold = ct.c_ulong(hold_time_ms)
        
        status = mcs.SA_GotoPositionRelative_S(
            self.mcsHandle,
            channel,
            shift,
            hold
        )
        self._exit_if_error(status)
        
        time.sleep(0.2)
    
    def get_position(self) -> int:
        """
        Get current position.
        
        Returns:
            Position in nanometers
        """
        channel = ct.c_ulong(self.axis_channel)
        position = ct.c_long()
        
        status = mcs.SA_GetPosition_S(
            self.mcsHandle,
            channel,
            position
        )
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
            start_nm: Start position (nanometers)
            end_nm: End position (nanometers)
            step_nm: Step size (nanometers)
            camera: Camera instance with acquire_single_image() method
            output_dir: Directory to save images and metadata
            settle_time_s: Wait time after each move (seconds)
            progress_callback: Optional function(current, total) for progress
        
        Returns:
            dict with sweep results:
                - positions: List of positions (nm)
                - image_files: List of saved image paths
                - metadata_file: Path to metadata JSON
                - start_time: ISO timestamp
                - duration_s: Total sweep time
        """
        print(f"[FilterStage] Starting sweep:")
        print(f"  Range: {start_nm} to {end_nm} nm ({(end_nm-start_nm)/1000:.3f} µm)")
        print(f"  Step: {step_nm} nm ({step_nm/1000:.3f} µm)")
        
        # Calculate positions
        positions = list(range(start_nm, end_nm + step_nm, step_nm))
        num_positions = len(positions)
        
        print(f"  Total positions: {num_positions}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        start_time = datetime.now()
        image_files = []
        actual_positions = []
        
        try:
            for idx, target_pos in enumerate(positions):
                # Move to position
                self.move_abs(target_pos)
                
                # Wait for settling
                time.sleep(settle_time_s)
                
                # Read back actual position
                actual_pos = self.get_position()
                actual_positions.append(actual_pos)
                
                # Capture image
                try:
                    image = camera.acquire_single_image()
                    
                    # Save image
                    image_filename = f"img_{idx:04d}_pos_{actual_pos}nm.tif"
                    image_path = output_path / image_filename
                    
                    # Save as 16-bit TIFF
                    import tifffile
                    tifffile.imwrite(str(image_path), image)
                    
                    image_files.append(str(image_path))
                    
                    print(f"  [{idx+1}/{num_positions}] "
                          f"Pos={actual_pos}nm ({actual_pos/1000:.3f}µm) "
                          f"→ {image_filename}")
                
                except Exception as e:
                    print(f"  ⚠️  Failed to capture at position {actual_pos}nm: {e}")
                    image_files.append(None)
                
                # Progress callback
                if progress_callback:
                    progress_callback(idx + 1, num_positions)
        
        except KeyboardInterrupt:
            print("\n[FilterStage] ⚠️  Sweep interrupted by user")
        
        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Save metadata
        metadata = {
            'sweep_config': {
                'start_nm': start_nm,
                'end_nm': end_nm,
                'step_nm': step_nm,
                'requested_positions': num_positions
            },
            'actual_data': {
                'target_positions_nm': positions,
                'actual_positions_nm': actual_positions,
                'image_files': image_files
            },
            'timing': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'settle_time_s': settle_time_s
            },
            'hardware': {
                'mcs_locator': self.locator,
                'axis_channel': self.axis_channel
            }
        }
        
        # Save metadata JSON
        import json
        metadata_file = output_path / "sweep_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[FilterStage] ✅ Sweep complete!")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Images saved: {len([f for f in image_files if f is not None])}/{num_positions}")
        print(f"  Output: {output_dir}")
        
        return {
            'positions': actual_positions,
            'image_files': image_files,
            'metadata_file': str(metadata_file),
            'start_time': start_time.isoformat(),
            'duration_s': duration
        }
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def home(self):
        """Move to zero position."""
        print("[FilterStage] Moving to home (0 nm)...")
        self.move_abs(0)
    
    def get_status(self) -> Dict:
        """Get stage status."""
        try:
            pos = self.get_position()
            
            # Query stage status
            channel = ct.c_ulong(self.axis_channel)
            status = ct.c_ulong()
            mcs.SA_GetStatus_S(self.mcsHandle, channel, status)
            
            status_names = {
                mcs.SA_STOPPED_STATUS: 'STOPPED',
                mcs.SA_STEPPING_STATUS: 'STEPPING',
                mcs.SA_SCANNING_STATUS: 'SCANNING',
                mcs.SA_HOLDING_STATUS: 'HOLDING',
                mcs.SA_TARGET_STATUS: 'TARGET',
                mcs.SA_MOVE_DELAY_STATUS: 'MOVE_DELAY',
                mcs.SA_CALIBRATING_STATUS: 'CALIBRATING',
                mcs.SA_FINDING_REF_STATUS: 'FINDING_REF'
            }
            
            status_name = status_names.get(status.value, f'UNKNOWN({status.value})')
            
            return {
                'connected': not self._closed,
                'position_nm': pos,
                'position_um': pos / 1000.0,
                'status': status_name,
                'locator': self.locator,
                'channel': self.axis_channel
            }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
    
    def print_status(self):
        """Print stage status to console."""
        status = self.get_status()
        
        print("\n" + "="*60)
        print("FilterStage Status")
        print("="*60)
        print(f"Locator:   {status.get('locator', 'N/A')}")
        print(f"Channel:   {status.get('channel', 'N/A')}")
        print(f"Connected: {status.get('connected', False)}")
        
        if status.get('connected'):
            print(f"Position:  {status['position_nm']} nm ({status['position_um']:.3f} µm)")
            print(f"Status:    {status.get('status', 'UNKNOWN')}")
        else:
            print(f"Error:     {status.get('error', 'Unknown')}")
        
        print("="*60 + "\n")
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def close(self):
        """Close MCS connection."""
        if self._closed:
            return
        
        try:
            status = mcs.SA_CloseSystem(self.mcsHandle)
            self._exit_if_error(status)
        finally:
            self._closed = True
            print("[FilterStage] Closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("FilterStage Test")
    print("="*70)
    
    # Note: Replace with actual MCS locator
    # This example assumes you've run multi_mcs_manager.py first
    
    try:
        # Create filter stage (use actual locator from discovery)
        stage = FilterStage(locator="usb:id:1234", axis_channel=0)
        
        # Print status
        stage.print_status()
        
        # Test movement
        print("\nTest: Basic movement")
        print("-"*70)
        
        print("Moving to 10 µm (10000 nm)...")
        stage.move_abs(10000)
        pos = stage.get_position()
        print(f"Actual position: {pos} nm ({pos/1000:.3f} µm)")
        
        print("\nMoving relative by +5 µm...")
        stage.move_rel(5000)
        pos = stage.get_position()
        print(f"New position: {pos} nm ({pos/1000:.3f} µm)")
        
        # Home
        print("\nReturning to home...")
        stage.home()
        
        stage.close()
        print("\n✅ Test complete!")
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()