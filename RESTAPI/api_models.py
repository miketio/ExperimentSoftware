# api_models.py
"""
Pydantic models for REST API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


# ========================================
# Request Models
# ========================================

class CommandRequest(BaseModel):
    """Generic command request."""
    command: str = Field(..., description="CLI command to execute", example="pos")


class MoveAbsoluteRequest(BaseModel):
    """Move stage to absolute position."""
    axis: Literal["x", "y", "z"] = Field(..., description="Axis to move")
    position: int = Field(..., description="Target position in nanometers", example=5000)


class MoveRelativeRequest(BaseModel):
    """Move stage relative to current position."""
    axis: Literal["x", "y", "z"] = Field(..., description="Axis to move")
    shift: int = Field(..., description="Shift in nanometers (can be negative)", example=500)


class AutofocusRequest(BaseModel):
    """Run autofocus scan."""
    axis: Literal["x", "y", "z"] = Field(default="x", description="Axis to scan")
    range: Optional[int] = Field(default=None, description="Scan range in nanometers", example=10000)
    step: Optional[int] = Field(default=None, description="Step size in nanometers", example=500)
    enable_plot: bool = Field(default=True, description="Show live plot during scan")


class ROIRequest(BaseModel):
    """Set camera region of interest."""
    left: Optional[int] = Field(default=None, description="Left coordinate")
    top: Optional[int] = Field(default=None, description="Top coordinate")
    width: Optional[int] = Field(default=None, description="Width in pixels")
    height: Optional[int] = Field(default=None, description="Height in pixels")


# ========================================
# Response Models
# ========================================

class StatusResponse(BaseModel):
    """System status response."""
    status: str = Field(..., description="Status message")
    message: Optional[str] = Field(default=None, description="Additional details")


class PositionResponse(BaseModel):
    """Stage position response."""
    x: int = Field(..., description="X position in nanometers")
    y: int = Field(..., description="Y position in nanometers")
    z: int = Field(..., description="Z position in nanometers")


class SingleAxisPositionResponse(BaseModel):
    """Single axis position response."""
    axis: str = Field(..., description="Axis name")
    position: int = Field(..., description="Position in nanometers")


class AutofocusResultResponse(BaseModel):
    """Autofocus scan results."""
    axis: str = Field(..., description="Scanned axis")
    best_position: int = Field(..., description="Optimal focus position in nanometers")
    best_metric: float = Field(..., description="Focus quality metric at best position")
    scan_complete: bool = Field(..., description="Whether scan completed successfully")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="System health status")
    camera_connected: bool = Field(..., description="Camera connection status")
    stage_connected: bool = Field(..., description="Stage connection status")
    threads_running: dict = Field(..., description="Status of each thread")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    command: Optional[str] = Field(default=None, description="Command that caused error")