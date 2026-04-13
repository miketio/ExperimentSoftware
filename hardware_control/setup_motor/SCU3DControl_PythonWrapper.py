# hardware_control/setup_motor/SCU3DControl_PythonWrapper.py
"""
Python ctypes wrapper for SCU3DControl.dll  (SmarAct HCU / SCU family).

Mirrors the style of MCSControl_PythonWrapper.py so the rest of the
codebase stays consistent.

SCU3DControl.dll must be on PATH or in the working directory.
It is supplied on the CD that came with the device (version ≥ 1.5.7).
"""

import ctypes as ct

# ---------------------------------------------------------------------------
# Load the DLL
# ---------------------------------------------------------------------------
SCU_lib = ct.cdll.LoadLibrary("SCU3DControl")

# ---------------------------------------------------------------------------
# Status / error return codes  (from SCU3DControl.h)
# ---------------------------------------------------------------------------
SA_OK                             = 0
SA_INITIALIZATION_ERROR           = 1
SA_NOT_INITIALIZED_ERROR          = 2
SA_NO_DEVICES_FOUND_ERROR         = 3
SA_TOO_MANY_DEVICES_ERROR         = 4
SA_INVALID_DEVICE_INDEX_ERROR     = 5
SA_INVALID_CHANNEL_INDEX_ERROR    = 6
SA_TRANSMIT_ERROR                 = 7
SA_WRITE_ERROR                    = 8
SA_INVALID_PARAMETER_ERROR        = 9
SA_READ_ERROR                     = 10
SA_INTERNAL_ERROR                 = 12
SA_WRONG_MODE_ERROR               = 13
SA_PROTOCOL_ERROR                 = 14
SA_TIMEOUT_ERROR                  = 15
SA_NOTIFICATION_ALREADY_SET_ERROR = 16
SA_ID_LIST_TOO_SMALL_ERROR        = 17
SA_DEVICE_ALREADY_ADDED_ERROR     = 18
SA_DEVICE_NOT_FOUND_ERROR         = 19
SA_INVALID_COMMAND_ERROR          = 128
SA_COMMAND_NOT_SUPPORTED_ERROR    = 129
SA_NO_SENSOR_PRESENT_ERROR        = 130
SA_WRONG_SENSOR_TYPE_ERROR        = 131
SA_END_STOP_REACHED_ERROR         = 132
SA_COMMAND_OVERRIDDEN_ERROR       = 133
SA_HV_RANGE_ERROR                 = 134
SA_TEMP_OVERHEAT_ERROR            = 135
SA_CALIBRATION_FAILED_ERROR       = 136
SA_REFERENCING_FAILED_ERROR       = 137
SA_NOT_PROCESSABLE_ERROR          = 138
SA_OTHER_ERROR                    = 255

# Channel status codes
SA_STOPPED_STATUS              = 0
SA_SETTING_AMPLITUDE_STATUS    = 1
SA_MOVING_STATUS               = 2
SA_TARGETING_STATUS            = 3
SA_HOLDING_STATUS              = 4
SA_CALIBRATING_STATUS          = 5
SA_MOVING_TO_REFERENCE_STATUS  = 6

# SA_InitDevices configuration flags
SA_SYNCHRONOUS_COMMUNICATION   = 0
SA_ASYNCHRONOUS_COMMUNICATION  = 1
SA_HARDWARE_RESET              = 2

# Movement directions
SA_BACKWARD_DIRECTION = 0
SA_FORWARD_DIRECTION  = 1

# Auto-zero flag (for SA_MoveToReference_S)
SA_NO_AUTO_ZERO = 0
SA_AUTO_ZERO    = 1

# Sensor presence
SA_NO_SENSOR_PRESENT = 0
SA_SENSOR_PRESENT    = 1

# Physical position known
SA_PHYSICAL_POSITION_UNKNOWN = 0
SA_PHYSICAL_POSITION_KNOWN   = 1

# ---------------------------------------------------------------------------
# Section I: Initialization Functions
# ---------------------------------------------------------------------------

def SA_GetDLLVersion(version: ct.c_uint) -> int:
    """Get the DLL version."""
    return SCU_lib.SA_GetDLLVersion(ct.byref(version))


def SA_InitDevices(configuration: int) -> int:
    """
    Initialize all connected SCU devices.

    Must be called before any other function.
    configuration: SA_SYNCHRONOUS_COMMUNICATION (0) or SA_ASYNCHRONOUS_COMMUNICATION (1)
    """
    return SCU_lib.SA_InitDevices(ct.c_uint(configuration))


def SA_ReleaseDevices() -> int:
    """Release all devices. Call on shutdown."""
    return SCU_lib.SA_ReleaseDevices()


def SA_GetNumberOfDevices(number: ct.c_uint) -> int:
    """Get number of initialized devices."""
    return SCU_lib.SA_GetNumberOfDevices(ct.byref(number))


def SA_GetDeviceID(deviceIndex: ct.c_uint, deviceId: ct.c_uint) -> int:
    """Get the hardware device ID for a given device index."""
    return SCU_lib.SA_GetDeviceID(deviceIndex, ct.byref(deviceId))


def SA_GetDeviceFirmwareVersion(deviceIndex: ct.c_uint, version: ct.c_uint) -> int:
    """Get firmware version for a given device index."""
    return SCU_lib.SA_GetDeviceFirmwareVersion(deviceIndex, ct.byref(version))


# ---------------------------------------------------------------------------
# Section IIa.1: Configuration Functions (Synchronous)
# ---------------------------------------------------------------------------

def SA_GetSensorPresent_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint, present: ct.c_uint) -> int:
    """Check if a sensor is present on the given channel."""
    return SCU_lib.SA_GetSensorPresent_S(deviceIndex, channelIndex, ct.byref(present))


def SA_GetSensorType_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint, sensorType: ct.c_uint) -> int:
    """Get the sensor type for a given channel."""
    return SCU_lib.SA_GetSensorType_S(deviceIndex, channelIndex, ct.byref(sensorType))


def SA_SetSensorType_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint, sensorType: int) -> int:
    """Set the sensor type for a given channel."""
    return SCU_lib.SA_SetSensorType_S(deviceIndex, channelIndex, ct.c_uint(sensorType))


def SA_SetZero_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint) -> int:
    """Set the current position as zero (requires sensor)."""
    return SCU_lib.SA_SetZero_S(deviceIndex, channelIndex)


def SA_GetPhysicalPositionKnown_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint, known: ct.c_uint) -> int:
    """Check if the physical position is known (referenced)."""
    return SCU_lib.SA_GetPhysicalPositionKnown_S(deviceIndex, channelIndex, ct.byref(known))


def SA_SetSafeDirection_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint, direction: int) -> int:
    """Set the safe movement direction (SA_FORWARD_DIRECTION or SA_BACKWARD_DIRECTION)."""
    return SCU_lib.SA_SetSafeDirection_S(deviceIndex, channelIndex, ct.c_uint(direction))


def SA_GetSafeDirection_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint, direction: ct.c_uint) -> int:
    """Get the safe movement direction."""
    return SCU_lib.SA_GetSafeDirection_S(deviceIndex, channelIndex, ct.byref(direction))


def SA_SetClosedLoopMaxFrequency_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint, frequency: int) -> int:
    """Set the maximum closed-loop step frequency (Hz)."""
    return SCU_lib.SA_SetClosedLoopMaxFrequency_S(deviceIndex, channelIndex, ct.c_uint(frequency))


def SA_GetClosedLoopMaxFrequency_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint, frequency: ct.c_uint) -> int:
    """Get the maximum closed-loop step frequency (Hz)."""
    return SCU_lib.SA_GetClosedLoopMaxFrequency_S(deviceIndex, channelIndex, ct.byref(frequency))


def SA_SetChannelProperty_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint, key: int, value: int) -> int:
    """Set a channel property."""
    return SCU_lib.SA_SetChannelProperty_S(deviceIndex, channelIndex, ct.c_int(key), ct.c_int(value))


def SA_GetChannelProperty_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint, key: int, value: ct.c_int) -> int:
    """Get a channel property."""
    return SCU_lib.SA_GetChannelProperty_S(deviceIndex, channelIndex, ct.c_int(key), ct.byref(value))


# ---------------------------------------------------------------------------
# Section IIa.2: Movement Control Functions (Synchronous)
# ---------------------------------------------------------------------------

def SA_MoveStep_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint,
                  steps: int, amplitude: int, frequency: int) -> int:
    """
    Open-loop step move.

    steps     : number of steps (signed; negative = backward)
    amplitude : drive voltage × 10  (e.g. 1000 = 100 V)
    frequency : step frequency in Hz (e.g. 1000 = 1 kHz)
    """
    return SCU_lib.SA_MoveStep_S(deviceIndex, channelIndex,
                                  ct.c_int(steps),
                                  ct.c_uint(amplitude),
                                  ct.c_uint(frequency))


def SA_MovePositionAbsolute_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint,
                               position: int, holdTime: int) -> int:
    """
    Closed-loop absolute move (requires sensor + referenced position).

    position : target position in nanometers
    holdTime : time to hold position after arrival in milliseconds (0 = no hold)
    """
    return SCU_lib.SA_MovePositionAbsolute_S(deviceIndex, channelIndex,
                                              ct.c_int(position),
                                              ct.c_uint(holdTime))


def SA_MovePositionRelative_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint,
                               diff: int, holdTime: int) -> int:
    """
    Closed-loop relative move (requires sensor).

    diff     : distance to move in nanometers (signed)
    holdTime : time to hold position after arrival in milliseconds
    """
    return SCU_lib.SA_MovePositionRelative_S(deviceIndex, channelIndex,
                                              ct.c_int(diff),
                                              ct.c_uint(holdTime))


def SA_SetAmplitude_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint, amplitude: int) -> int:
    """Set the open-loop step amplitude (voltage × 10)."""
    return SCU_lib.SA_SetAmplitude_S(deviceIndex, channelIndex, ct.c_uint(amplitude))


def SA_CalibrateSensor_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint) -> int:
    """Calibrate the position sensor."""
    return SCU_lib.SA_CalibrateSensor_S(deviceIndex, channelIndex)


def SA_MoveToReference_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint,
                          holdTime: int, autoZero: int) -> int:
    """
    Move to reference mark to establish absolute position.

    holdTime : hold time after reaching reference in milliseconds
    autoZero : SA_AUTO_ZERO (1) to set position to 0 at reference,
               SA_NO_AUTO_ZERO (0) to keep the calibrated offset
    """
    return SCU_lib.SA_MoveToReference_S(deviceIndex, channelIndex,
                                         ct.c_uint(holdTime),
                                         ct.c_uint(autoZero))


def SA_MoveToEndStop_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint,
                        direction: int, holdTime: int, autoZero: int) -> int:
    """Move to the end stop in a given direction."""
    return SCU_lib.SA_MoveToEndStop_S(deviceIndex, channelIndex,
                                       ct.c_uint(direction),
                                       ct.c_uint(holdTime),
                                       ct.c_uint(autoZero))


def SA_Stop_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint) -> int:
    """Stop all motion on a channel immediately."""
    return SCU_lib.SA_Stop_S(deviceIndex, channelIndex)


# ---------------------------------------------------------------------------
# Section IIa.3: Channel Feedback Functions (Synchronous)
# ---------------------------------------------------------------------------

def SA_GetStatus_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint, status: ct.c_uint) -> int:
    """Get the current channel status code."""
    return SCU_lib.SA_GetStatus_S(deviceIndex, channelIndex, ct.byref(status))


def SA_GetPosition_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint, position: ct.c_int) -> int:
    """Get the current position in nanometers (requires sensor)."""
    return SCU_lib.SA_GetPosition_S(deviceIndex, channelIndex, ct.byref(position))


def SA_GetAmplitude_S(deviceIndex: ct.c_uint, channelIndex: ct.c_uint, amplitude: ct.c_uint) -> int:
    """Get the current step amplitude."""
    return SCU_lib.SA_GetAmplitude_S(deviceIndex, channelIndex, ct.byref(amplitude))