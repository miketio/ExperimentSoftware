# smaract_stage.py
import ctypes as ct
import time
from typing import Dict, Optional

from hardware_control.setup_motor.xyz_stage_base import XYZStageBase

# Import your MCS wrapper here; name must match your environment
import hardware_control.setup_motor.MCSControl_PythonWrapper as mcs


class SmarActXYZStage(XYZStageBase):
    """
    Concrete implementation for the SmarAct MCS controller.
    Notes:
      - Keeps the axis mapping configurable (default matches your original code).
      - Positions and shifts are in nanometers (int).
      - Raises RuntimeError on MCS errors with helpful message.
    """

    def __init__(self, axis_map: Dict[str, int] | None = None, find_buffer_size: int = 256):
        # handle & initialization
        self.mcsHandle = ct.c_ulong()
        self._closed = False

        # axis name -> channel index mapping (default from your original)
        self.axis_dict = axis_map or {'x': 0, 'y': 1, 'z': 2}

        # helper buffer sizes
        buffer_size = max(256, find_buffer_size)
        outBuffer = ct.create_string_buffer(buffer_size)
        ioBufferSize = ct.c_ulong(buffer_size)

        # Try to find systems
        # NOTE: MCSControl_PythonWrapper expects a Python str for the options argument
        status = mcs.SA_FindSystems('', outBuffer, ioBufferSize)
        self._exit_if_error(status)

        # decode the returned string (value up to first null byte)
        found = outBuffer.value.decode('utf-8', errors='ignore')
        if not found:
            raise RuntimeError("No MCS systems found by SA_FindSystems().")
        print(f"MCS address: {found}")

        # open system in sync mode (reset) — pass a Python str, not bytes
        status = mcs.SA_OpenSystem(self.mcsHandle, outBuffer, "sync,reset")
        self._exit_if_error(status)

        # get number of channels (optional print)
        num_channels = ct.c_ulong()
        status = mcs.SA_GetNumberOfChannels(self.mcsHandle, num_channels)
        self._exit_if_error(status)
        print(f"Number of Channels: {num_channels.value}")

        # check sensor type on channel 0 (we'll just query one channel)
        sensorType = ct.c_ulong()
        status = mcs.SA_GetSensorType_S(self.mcsHandle, ct.c_ulong(0), sensorType)
        self._exit_if_error(status)
        print(f"Sensor type for channel 0: {sensorType.value}")

        # optional: check physical position known for channels 0,1,2
        for ch in (0, 1, 2):
            known = ct.c_uint()
            try:
                result = mcs.SA_GetPhysicalPositionKnown_S(self.mcsHandle, ct.c_ulong(ch), known)
                if result == mcs.SA_OK:
                    print(f"Position in channel {ch} OK")
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
                # many wrappers accept (status, buffer)
                mcs.SA_GetStatusInfo(ct.c_ulong(status), err_buf)
                msg = err_buf.value.decode('utf-8', errors='ignore')
            except Exception:
                msg = f"MCS error code {status}"
        except Exception:
            msg = f"MCS error code {status}"
        raise RuntimeError(msg)

    def move_abs(self, axis: str, pos: int) -> None:
        """Absolute move: pos in nm"""
        if axis not in self.axis_dict:
            raise ValueError(f"Unknown axis '{axis}'")
        channel = ct.c_ulong(self.axis_dict[axis])
        status = mcs.SA_GotoPositionAbsolute_S(self.mcsHandle, channel, ct.c_long(pos), ct.c_ulong(0))
        self._exit_if_error(status)
        time.sleep(0.2)

    def move_rel(self, axis: str, shift: int, sleep_time: float = 0.01) -> None:
        """Relative move: shift in nm"""
        if axis not in self.axis_dict:
            raise ValueError(f"Unknown axis '{axis}'")
        channel = ct.c_ulong(self.axis_dict[axis])
        status = mcs.SA_GotoPositionRelative_S(self.mcsHandle, channel, ct.c_long(shift), ct.c_ulong(1))
        self._exit_if_error(status)
        time.sleep(sleep_time)

    def get_pos(self, axis: str) -> int:
        """Readback position (nm)"""
        if axis not in self.axis_dict:
            raise ValueError(f"Unknown axis '{axis}'")
        channel = ct.c_ulong(self.axis_dict[axis])
        position = ct.c_long()
        # some wrapper variants expect the third argument byref, others by value; adjust if needed
        status = mcs.SA_GetPosition_S(self.mcsHandle, channel, position)
        self._exit_if_error(status)
        return int(position.value)

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        status = mcs.SA_CloseSystem(self.mcsHandle)
        self._exit_if_error(status)
        self._closed = True
        print("SmarAct 3D stage closed.")


if __name__ == "__main__":
    """
    Quick smoke test when running this file directly:
      - tries to connect to the first MCS system
      - prints the readback position for x,y,z (if available)
      - closes the system
    """
    stage = None
    try:
        print("Attempting to initialize SmarAct stage...")
        stage = SmarActXYZStage()
        print("Initialization successful.\nQuerying positions...")

        for ax in ("x", "y", "z"):
            try:
                pos = stage.get_pos(ax)
                print(f"Axis {ax}: {pos} nm")
            except Exception as e:
                print(f"  Could not read axis {ax}: {e}")

        # Optional: small relative move test (uncomment if you want a movement test)
        # print("Doing a small relative move on X...")
        # stage.move_rel('x', 100000)  # move by 100 µm (100000 nm)
        # print("New X pos:", stage.get_pos('x'))

    except Exception as e:
        print("Failed to initialize or communicate with SmarAct stage:")
        print(" ", repr(e))
    finally:
        if stage is not None:
            try:
                stage.close()
            except Exception as e:
                print("Error while closing stage:", e)
        print("Done.")
