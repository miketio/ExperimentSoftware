# hardware_control/setup_motor/multi_mcs_manager.py
"""
Multi-MCS Manager - Handle multiple SmarAct MCS controllers

Discovers all connected MCS devices and allows assignment to different roles:
- XYZ stage (3 axes for sample positioning)
- K-space filter stage (1 axis for spectral filtering)

Enhanced with:
- Automatic role detection based on channel count
- 10s initialization delay for 1D filter stage
- Validation of device assignments
"""

import ctypes as ct
import time
from typing import List, Dict, Optional, Tuple
import hardware_control.setup_motor.MCSControl_PythonWrapper as mcs


class MCSDeviceInfo:
    """Information about a discovered MCS device."""
    
    def __init__(self, locator: str, index: int):
        self.locator = locator
        self.index = index
        self.handle = None
        self.num_channels = 0
        self.is_open = False
        self.role = None  # 'xyz_stage' or 'filter_stage'
    
    def __repr__(self):
        return f"MCS[{self.index}](locator={self.locator}, channels={self.num_channels}, role={self.role})"


class MultiMCSManager:
    """
    Discovers and manages multiple MCS controllers.
    
    Usage:
        manager = MultiMCSManager()
        devices = manager.discover_devices()
        
        # Auto-assign based on channel count
        manager.auto_assign_roles()
        
        # Or manually assign
        manager.assign_device(0, role='xyz_stage')
        manager.assign_device(1, role='filter_stage')
        
        # Get stage instances
        xyz_stage = manager.get_xyz_stage()
        filter_stage = manager.get_filter_stage()
    """
    
    def __init__(self, buffer_size: int = 1024):
        self.buffer_size = buffer_size
        self.devices: List[MCSDeviceInfo] = []
        self.xyz_stage = None
        self.filter_stage = None
    
    def discover_devices(self) -> List[MCSDeviceInfo]:
        """
        Discover all connected MCS devices.
        
        Returns:
            List of MCSDeviceInfo objects
        """
        print("[MultiMCSManager] Discovering MCS devices...")
        
        # Create buffer for device locators
        out_buffer = ct.create_string_buffer(2048)
        io_buffer_size = ct.c_ulong(2048)
        
        # Find all systems
        status = mcs.SA_FindSystems('', out_buffer, io_buffer_size)
        if status != mcs.SA_OK:
            if status == mcs.SA_NO_SYSTEMS_FOUND_ERROR:
                print("[MultiMCSManager] ⚠️  No MCS devices found (SA_NO_SYSTEMS_FOUND).")
                return []
            error_msg = self._get_error_message(status)
            raise RuntimeError(f"SA_FindSystems failed: {error_msg}")
        
        # Decode locator string
        locator_string = out_buffer.value.decode('utf-8', errors='ignore').strip()
        
        if not locator_string:
            print("[MultiMCSManager] ⚠️  No MCS devices found (Empty string)!")
            return []
        
        # Handle newline delimiters which MCS uses
        cleaned_string = locator_string.replace('\n', ',').replace('\r', ',')
        locators = [loc.strip() for loc in cleaned_string.split(',') if loc.strip()]
        
        print(f"[MultiMCSManager] Found {len(locators)} MCS device(s):")
        
        # Create device info for each
        self.devices = []
        for idx, locator in enumerate(locators):
            device = MCSDeviceInfo(locator, idx)
            
            # Open device to query info
            try:
                self._open_device(device)
                self._query_device_info(device)
                self._close_device(device)
                
                # Only add if successfully queried
                self.devices.append(device)
                print(f"  ✅ MCS[{idx}]: {locator} ({device.num_channels} channels)")
                
            except Exception as e:
                print(f"  ⚠️  MCS[{idx}] ({locator}): Failed to query - {e}")
                self._close_device(device)
                continue
        
        return self.devices

    def _open_device(self, device: MCSDeviceInfo, options: str = "sync"):
        """Open a device connection."""
        device.handle = ct.c_ulong()
        
        # Convert locator to bytes for ctypes
        locator_bytes = device.locator.encode('utf-8')
        options_bytes = options.encode('utf-8')
        
        status = mcs.SA_OpenSystem(
            device.handle,
            locator_bytes,
            options_bytes
        )
        
        if status != mcs.SA_OK:
            raise RuntimeError(f"Failed to open device: {self._get_error_message(status)}")
        
        device.is_open = True
    
    def _close_device(self, device: MCSDeviceInfo):
        """Close a device connection."""
        if device.is_open and device.handle:
            try:
                mcs.SA_CloseSystem(device.handle)
                device.is_open = False
            except Exception as e:
                print(f"  ⚠️  Error closing device: {e}")
    
    def _query_device_info(self, device: MCSDeviceInfo):
        """Query number of channels from device."""
        num_channels = ct.c_ulong()
        status = mcs.SA_GetNumberOfChannels(device.handle, num_channels)
        
        if status != mcs.SA_OK:
            raise RuntimeError(f"Failed to get channel count: {self._get_error_message(status)}")
        
        device.num_channels = num_channels.value
    
    def auto_assign_roles(self) -> bool:
        """
        Automatically assign roles based on channel count.
        
        Logic:
        - 3-channel device → XYZ stage
        - 1-channel device → Filter stage
        
        Returns:
            True if assignment successful, False otherwise
        """
        print("\n[MultiMCSManager] Auto-assigning roles based on channel count...")
        
        xyz_candidates = [d for d in self.devices if d.num_channels == 3]
        filter_candidates = [d for d in self.devices if d.num_channels == 1]
        
        # Assign XYZ stage (3 channels)
        if xyz_candidates:
            xyz_device = xyz_candidates[0]
            xyz_device.role = 'xyz_stage'
            print(f"  ✅ MCS[{xyz_device.index}] ({xyz_device.locator}) → XYZ Stage (3 channels)")
        else:
            print(f"  ⚠️  No 3-channel device found for XYZ stage!")
            return False
        
        # Assign filter stage (1 channel)
        if filter_candidates:
            filter_device = filter_candidates[0]
            filter_device.role = 'filter_stage'
            print(f"  ✅ MCS[{filter_device.index}] ({filter_device.locator}) → Filter Stage (1 channel)")
        else:
            print(f"  ⚠️  No 1-channel device found for filter stage")
            print(f"  → System will operate with XYZ stage only")
        
        return True
    
    def assign_device(self, device_index: int, role: str):
        """
        Manually assign a role to a discovered device.
        
        Args:
            device_index: Index from discover_devices() list
            role: Either 'xyz_stage' or 'filter_stage'
        """
        if device_index >= len(self.devices):
            raise ValueError(f"Invalid device index {device_index} (only {len(self.devices)} devices found)")
        
        if role not in ['xyz_stage', 'filter_stage']:
            raise ValueError(f"Invalid role '{role}' (must be 'xyz_stage' or 'filter_stage')")
        
        device = self.devices[device_index]
        
        # Validate channel count for role
        if role == 'xyz_stage' and device.num_channels != 3:
            print(f"[MultiMCSManager] ⚠️ WARNING: Assigning {device.num_channels}-channel device as XYZ stage")
            print(f"[MultiMCSManager]   Expected 3 channels. This may not work correctly!")
        
        if role == 'filter_stage' and device.num_channels != 1:
            print(f"[MultiMCSManager] ⚠️ WARNING: Assigning {device.num_channels}-channel device as filter stage")
            print(f"[MultiMCSManager]   Expected 1 channel. Only first channel will be used.")
        
        device.role = role
        print(f"[MultiMCSManager] ✅ Assigned MCS[{device_index}] ({device.locator}) as {role}")
    
    def get_xyz_stage(self, axis_map: Optional[Dict[str, int]] = None):
        """
        Get SmarActXYZStage instance for the XYZ stage controller.
        
        Args:
            axis_map: Optional axis mapping (default: {'x': 0, 'y': 1, 'z': 2})
        
        Returns:
            SmarActXYZStage instance
        """
        from hardware_control.setup_motor.smartact_stage import SmarActXYZStage
        
        # Find device with xyz_stage role
        xyz_device = next((d for d in self.devices if d.role == 'xyz_stage'), None)
        
        if xyz_device is None:
            raise RuntimeError("No device assigned as 'xyz_stage'. Call assign_device() or auto_assign_roles() first.")
        
        # Validate channel count
        if xyz_device.num_channels != 3:
            print(f"[MultiMCSManager] ⚠️ WARNING: XYZ stage has {xyz_device.num_channels} channels (expected 3)")
        
        # Create stage instance with specific locator
        print(f"[MultiMCSManager] Initializing XYZ stage...")
        self.xyz_stage = SmarActXYZStage(
            locator=xyz_device.locator,
            axis_map=axis_map,
            options="sync"
        )
        
        print(f"[MultiMCSManager] ✅ Created XYZ stage from MCS[{xyz_device.index}]")
        return self.xyz_stage
    
    def get_filter_stage(self, axis_channel: int = 0):
        """
        Get FilterStage instance for the K-space filter controller.
        
        IMPORTANT: 1D filter stage requires ~10 seconds to initialize!
        
        Args:
            axis_channel: Channel index for filter axis (default: 0)
        
        Returns:
            FilterStage instance
        """
        from hardware_control.setup_motor.filter_stage import FilterStage
        
        # Find device with filter_stage role
        filter_device = next((d for d in self.devices if d.role == 'filter_stage'), None)
        
        if filter_device is None:
            raise RuntimeError("No device assigned as 'filter_stage'. Call assign_device() or auto_assign_roles() first.")
        
        # Validate channel count
        if filter_device.num_channels != 1:
            print(f"[MultiMCSManager] ⚠️ WARNING: Filter stage has {filter_device.num_channels} channels (expected 1)")
            print(f"[MultiMCSManager]   Using channel {axis_channel} only")
        
        # Create filter stage with 10s initialization delay
        print(f"[MultiMCSManager] Initializing filter stage...")
        print(f"[MultiMCSManager] ⏳ Please wait ~10 seconds for 1D stage initialization...")
        
        start_time = time.time()
        self.filter_stage = FilterStage(
            locator=filter_device.locator,
            axis_channel=axis_channel
        )
        elapsed = time.time() - start_time
        
        print(f"[MultiMCSManager] ✅ Created Filter stage from MCS[{filter_device.index}] (took {elapsed:.1f}s)")
        return self.filter_stage
    
    def _get_error_message(self, status: int) -> str:
        """Get human-readable error message."""
        try:
            err_buf = ct.create_string_buffer(256)
            mcs.SA_GetStatusInfo(ct.c_ulong(status), err_buf)
            return err_buf.value.decode('utf-8', errors='ignore')
        except:
            return f"Error code {status}"
    
    def validate_assignments(self) -> bool:
        """
        Validate that device assignments are correct.
        
        Returns:
            True if valid, False otherwise
        """
        print("\n[MultiMCSManager] Validating device assignments...")
        
        xyz_devices = [d for d in self.devices if d.role == 'xyz_stage']
        filter_devices = [d for d in self.devices if d.role == 'filter_stage']
        
        is_valid = True
        
        # Check XYZ stage
        if not xyz_devices:
            print("  ❌ No device assigned as XYZ stage!")
            is_valid = False
        elif len(xyz_devices) > 1:
            print(f"  ⚠️  Multiple devices assigned as XYZ stage ({len(xyz_devices)})")
            is_valid = False
        elif xyz_devices[0].num_channels != 3:
            print(f"  ⚠️  XYZ stage has {xyz_devices[0].num_channels} channels (expected 3)")
            is_valid = False
        else:
            print(f"  ✅ XYZ stage: MCS[{xyz_devices[0].index}] with 3 channels")
        
        # Check filter stage
        if not filter_devices:
            print("  ℹ️  No filter stage assigned (optional)")
        elif len(filter_devices) > 1:
            print(f"  ⚠️  Multiple devices assigned as filter stage ({len(filter_devices)})")
            is_valid = False
        elif filter_devices[0].num_channels != 1:
            print(f"  ⚠️  Filter stage has {filter_devices[0].num_channels} channels (expected 1)")
        else:
            print(f"  ✅ Filter stage: MCS[{filter_devices[0].index}] with 1 channel")
        
        return is_valid
    
    def print_device_summary(self):
        """Print summary of all discovered devices."""
        print("\n" + "="*70)
        print("MCS Device Summary")
        print("="*70)
        
        if not self.devices:
            print("No devices found.")
            return
        
        for device in self.devices:
            role_str = device.role if device.role else "Not assigned"
            print(f"\nMCS[{device.index}]:")
            print(f"  Locator:  {device.locator}")
            print(f"  Channels: {device.num_channels}")
            print(f"  Role:     {role_str}")
            
            # Suggest role based on channels
            if not device.role:
                if device.num_channels == 3:
                    print(f"  Suggestion: xyz_stage (3-axis positioning)")
                elif device.num_channels == 1:
                    print(f"  Suggestion: filter_stage (1-axis filtering)")
        
        print("="*70 + "\n")

    def close_all(self):
        """
        Close all MCS devices and stages.
        
        Call this during application shutdown to properly cleanup all hardware.
        """
        print("[MultiMCSManager] Closing all devices...")
        
        # Close XYZ stage
        if self.xyz_stage is not None:
            try:
                self.xyz_stage.close()
                self.xyz_stage = None
                print("[MultiMCSManager]   ✅ XYZ stage closed")
            except Exception as e:
                print(f"[MultiMCSManager]   ❌ Error closing XYZ stage: {e}")
        
        # Close filter stage
        if self.filter_stage is not None:
            try:
                self.filter_stage.close()
                self.filter_stage = None
                print("[MultiMCSManager]   ✅ Filter stage closed")
            except Exception as e:
                print(f"[MultiMCSManager]   ❌ Error closing filter stage: {e}")
        
        # Close any open device handles
        for device in self.devices:
            try:
                self._close_device(device)
            except Exception as e:
                print(f"[MultiMCSManager]   ⚠️  Error closing device {device.index}: {e}")
        
        print("[MultiMCSManager] All devices closed")
# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Example: Discover and assign roles to multiple MCS controllers."""
    
    print("Multi-MCS Manager Example (Enhanced)")
    print("="*70)
    
    # Step 1: Discover all devices
    manager = MultiMCSManager()
    devices = manager.discover_devices()
    
    if len(devices) == 0:
        print("❌ No MCS devices found!")
        return
    
    # Step 2: Show devices
    manager.print_device_summary()
    
    # Step 3: Auto-assign roles based on channel count
    if not manager.auto_assign_roles():
        print("❌ Failed to auto-assign roles!")
        return
    
    # Step 4: Validate assignments
    if not manager.validate_assignments():
        print("⚠️  Device assignment validation failed!")
        print("    Proceeding anyway...")
    
    # Step 5: Create XYZ stage instance
    try:
        xyz_stage = manager.get_xyz_stage()
        print(f"\n✅ XYZ Stage ready: {xyz_stage}")
    except Exception as e:
        print(f"❌ Failed to create XYZ stage: {e}")
        return
    
    # Step 6: Create filter stage instance (with 10s wait)
    try:
        filter_stage = manager.get_filter_stage()
        print(f"✅ Filter Stage ready: {filter_stage}")
    except RuntimeError as e:
        print(f"ℹ️  No filter stage available: {e}")
        filter_stage = None
    
    # Step 7: Test XYZ stage
    print("\n" + "="*70)
    print("Testing XYZ Stage")
    print("="*70)
    
    for axis in ['x', 'y', 'z']:
        try:
            pos = xyz_stage.get_pos(axis)
            print(f"  {axis.upper()}: {pos} nm ({pos/1000:.3f} µm)")
        except Exception as e:
            print(f"  {axis.upper()}: Error - {e}")
    
    # Step 8: Test Filter stage (if present)
    if filter_stage:
        print("\n" + "="*70)
        print("Testing Filter Stage")
        print("="*70)
        
        try:
            pos = filter_stage.get_position()
            print(f"  Filter position: {pos} nm ({pos/1000:.3f} µm)")
            
            # Optional: Test small movement
            print("\n  Testing movement to 1000 nm...")
            filter_stage.move_abs(1000)
            new_pos = filter_stage.get_position()
            print(f"  New position: {new_pos} nm ({new_pos/1000:.3f} µm)")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Cleanup
    print("\n" + "="*70)
    print("Cleanup")
    print("="*70)
    xyz_stage.close()
    if filter_stage:
        filter_stage.close()
    
    print("\n✅ Example complete!")


if __name__ == "__main__":
    example_usage()