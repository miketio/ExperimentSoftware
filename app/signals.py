"""
System-wide Qt Signals

Centralized signal hub for loose coupling between components.
All application-wide events flow through this class.
"""

from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np
from typing import Dict, Any


class SystemSignals(QObject):
    """
    Global signal hub for the application.
    
    Naming convention:
    - Events: past tense (e.g., camera_connected)
    - Requests: present tense (e.g., request_camera_frame)
    - Progress: _progress suffix
    - Completion: _complete suffix
    """
    
    # ========================================================================
    # Hardware Signals
    # ========================================================================
    
    # Camera
    camera_connected = pyqtSignal(bool)  # True if connected
    camera_disconnected = pyqtSignal()
    camera_error = pyqtSignal(str)  # Error message
    
    # Stage
    stage_connected = pyqtSignal(bool)  # True if connected
    stage_disconnected = pyqtSignal()
    stage_position_changed = pyqtSignal(str, float)  # axis, position (µm)
    stage_move_complete = pyqtSignal()
    stage_error = pyqtSignal(str)  # Error message
    
    # Hardware mode
    hardware_mode_changed = pyqtSignal(str)  # "mock" or "real"
    
    # ========================================================================
    # Camera Stream Signals
    # ========================================================================
    
    frame_ready = pyqtSignal(object)  # numpy array (8-bit RGB or grayscale)
    frame_dropped = pyqtSignal()  # Frame skipped due to processing
    
    # Color scale
    color_scale_changed = pyqtSignal()
    colormap_changed = pyqtSignal(str)  # colormap name
    
    # ✅ NEW: Camera stream control
    request_stop_camera_stream = pyqtSignal()
    request_start_camera_stream = pyqtSignal()
    camera_stream_stopped = pyqtSignal()
    camera_stream_started = pyqtSignal()
    
    # ========================================================================
    # Alignment Signals
    # ========================================================================
    
    # Global alignment
    global_alignment_started = pyqtSignal()
    global_alignment_progress = pyqtSignal(int, str, object)  # block_id, status, thumbnail
    global_alignment_complete = pyqtSignal(dict)  # results dict
    global_alignment_failed = pyqtSignal(str)  # error message
    global_alignment_cancelled = pyqtSignal()
    
    # Block alignment
    block_alignment_started = pyqtSignal(int)  # block_id
    block_alignment_progress = pyqtSignal(int, str)  # block_id, status
    block_alignment_complete = pyqtSignal(int, dict)  # block_id, results
    block_alignment_failed = pyqtSignal(int, str)  # block_id, error
    
    # Batch alignment
    batch_alignment_started = pyqtSignal(list)  # list of block_ids
    batch_alignment_block_complete = pyqtSignal(int, dict)  # block_id, results
    batch_alignment_complete = pyqtSignal(dict)  # summary results
    
    # ========================================================================
    # Navigation Signals
    # ========================================================================
    
    # Block selection
    block_selected = pyqtSignal(int)  # block_id
    block_deselected = pyqtSignal()
    
    # Waveguide navigation
    navigation_started = pyqtSignal(int, int, str)  # block_id, waveguide, grating_side
    navigation_progress = pyqtSignal(str)  # status message
    navigation_complete = pyqtSignal()
    navigation_failed = pyqtSignal(str)  # error message
    
    # Target setting
    target_waveguide_changed = pyqtSignal(int)  # waveguide number
    target_grating_changed = pyqtSignal(str)  # "left", "center", "right"
    
    # ========================================================================
    # Autofocus Signals
    # ========================================================================
    
    autofocus_started = pyqtSignal(str)  # axis ("x", "y", "z")
    autofocus_progress = pyqtSignal(float, float, float)  # position, metric, progress%
    autofocus_complete = pyqtSignal(float, float)  # best_position (µm), best_metric
    autofocus_failed = pyqtSignal(str)  # error message
    autofocus_cancelled = pyqtSignal()
    
    # ========================================================================
    # UI State Signals
    # ========================================================================
    
    # Status messages
    status_message = pyqtSignal(str)  # Temporary message
    status_message_persistent = pyqtSignal(str)  # Stays until changed
    error_occurred = pyqtSignal(str, str)  # title, message
    warning_occurred = pyqtSignal(str, str)  # title, message
    
    # Busy state
    busy_started = pyqtSignal(str)  # operation name
    busy_ended = pyqtSignal()
    
    # Progress dialogs
    show_progress_dialog = pyqtSignal(str, bool)  # title, cancellable
    update_progress_dialog = pyqtSignal(int, str)  # percent, message
    close_progress_dialog = pyqtSignal()
    
    # ========================================================================
    # State Change Signals
    # ========================================================================
    
    state_loaded = pyqtSignal(str)  # filename
    state_saved = pyqtSignal(str)  # filename
    state_reset = pyqtSignal()
    
    # Configuration
    config_changed = pyqtSignal(str)  # config section name
    
    # ========================================================================
    # Export/Log Signals
    # ========================================================================
    
    export_started = pyqtSignal(str)  # export type
    export_complete = pyqtSignal(str)  # filename
    export_failed = pyqtSignal(str)  # error message
    
    log_message = pyqtSignal(str, str)  # level, message
    
    # ========================================================================
    # Request Signals (for inter-widget communication)
    # ========================================================================
    
    request_camera_frame = pyqtSignal()  # Request single frame
    request_stage_position = pyqtSignal()  # Request position update
    request_alignment_status = pyqtSignal()  # Request status refresh
    
    # ========================================================================
    # Shutdown Signal
    # ========================================================================
    
    application_closing = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        # Optional: Track signal connections for debugging
        self._connection_count = {}
    
    def emit_status(self, message: str, persistent: bool = False):
        """Convenience method for status messages."""
        if persistent:
            self.status_message_persistent.emit(message)
        else:
            self.status_message.emit(message)
    
    def emit_error(self, title: str, message: str):
        """Convenience method for errors."""
        self.error_occurred.emit(title, message)
        self.log_message.emit("ERROR", f"{title}: {message}")
    
    def emit_warning(self, title: str, message: str):
        """Convenience method for warnings."""
        self.warning_occurred.emit(title, message)
        self.log_message.emit("WARNING", f"{title}: {message}")
    
    def emit_log(self, level: str, message: str):
        """Convenience method for logging."""
        self.log_message.emit(level.upper(), message)