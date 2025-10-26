# api_server.py
"""
FastAPI REST API server for experiment control.
Provides HTTP endpoints to control camera, stage, and autofocus system.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import threading
import time
from typing import Optional

from api_models import (
    CommandRequest,
    MoveAbsoluteRequest,
    MoveRelativeRequest,
    AutofocusRequest,
    ROIRequest,
    StatusResponse,
    PositionResponse,
    SingleAxisPositionResponse,
    AutofocusResultResponse,
    HealthResponse,
    ErrorResponse,
)


class ExperimentAPI:
    """FastAPI wrapper for experiment control system."""
    
    def __init__(self, app_instance):
        """
        Initialize API with reference to DualThreadApp instance.
        
        Args:
            app_instance: Instance of DualThreadApp with camera, stage, autofocus
        """
        self.app_instance = app_instance
        self.fastapi_app = FastAPI(
            title="Microscopy Experiment Control API",
            description="REST API for controlling Andor camera and SmarAct stage",
            version="1.0.0",
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup all API routes."""
        app = self.fastapi_app
        
        # ========================================
        # System Endpoints
        # ========================================
        
        @app.get("/", response_model=StatusResponse)
        async def root():
            """API root endpoint."""
            return StatusResponse(
                status="ok",
                message="Microscopy Experiment Control API is running"
            )
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Check system health and component status."""
            try:
                camera_ok = self.app_instance.camera is not None
                stage_ok = self.app_instance.stage is not None
                
                threads_status = {
                    "camera": self.app_instance.camera_thread.is_alive() if self.app_instance.camera_thread else False,
                    "stage": self.app_instance.stage_thread.is_alive() if self.app_instance.stage_thread else False,
                    "input": self.app_instance.input_thread.is_alive() if self.app_instance.input_thread else False,
                }
                
                return HealthResponse(
                    status="healthy",
                    camera_connected=camera_ok,
                    stage_connected=stage_ok,
                    threads_running=threads_status
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
        
        # ========================================
        # Command Execution
        # ========================================
        
        @app.post("/command", response_model=StatusResponse)
        async def execute_command(request: CommandRequest):
            """
            Execute a CLI command.
            
            This endpoint accepts any CLI command string and executes it
            through the existing command processing system.
            """
            try:
                command = request.command.strip()
                if not command:
                    raise HTTPException(status_code=400, detail="Command cannot be empty")
                
                # Add to command queue
                with self.app_instance.queue_lock:
                    self.app_instance.command_queue.append(command)
                
                # Wait a bit for execution (simple approach)
                # For production, implement proper async result tracking
                time.sleep(0.5)
                
                return StatusResponse(
                    status="ok",
                    message=f"Command '{command}' executed"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # ========================================
        # Position & Status
        # ========================================
        
        @app.get("/status", response_model=PositionResponse)
        async def get_status():
            """Get current position of all axes."""
            try:
                x = self.app_instance.stage_app.get_pos('x')
                y = self.app_instance.stage_app.get_pos('y')
                z = self.app_instance.stage_app.get_pos('z')
                
                return PositionResponse(x=x, y=y, z=z)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")
        
        @app.get("/position/{axis}", response_model=SingleAxisPositionResponse)
        async def get_position(axis: str):
            """Get current position of specific axis."""
            axis = axis.lower()
            if axis not in ['x', 'y', 'z']:
                raise HTTPException(status_code=400, detail=f"Invalid axis '{axis}'. Must be x, y, or z.")
            
            try:
                position = self.app_instance.stage_app.get_pos(axis)
                return SingleAxisPositionResponse(axis=axis, position=position)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get position: {str(e)}")
        
        # ========================================
        # Movement Control
        # ========================================
        
        @app.post("/move/absolute", response_model=SingleAxisPositionResponse)
        async def move_absolute(request: MoveAbsoluteRequest):
            """Move stage axis to absolute position."""
            try:
                self.app_instance.stage_app.move_abs(request.axis, request.position)
                
                # Get actual position after move
                actual_position = self.app_instance.stage_app.get_pos(request.axis)
                
                return SingleAxisPositionResponse(
                    axis=request.axis,
                    position=actual_position
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to move {request.axis} to {request.position}nm: {str(e)}"
                )
        
        @app.post("/move/relative", response_model=SingleAxisPositionResponse)
        async def move_relative(request: MoveRelativeRequest):
            """Move stage axis relative to current position."""
            try:
                self.app_instance.stage_app.move_rel(request.axis, request.shift)
                
                # Get actual position after move
                actual_position = self.app_instance.stage_app.get_pos(request.axis)
                
                return SingleAxisPositionResponse(
                    axis=request.axis,
                    position=actual_position
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to move {request.axis} by {request.shift}nm: {str(e)}"
                )
        
        # ========================================
        # Autofocus
        # ========================================
        
        @app.post("/autofocus", response_model=AutofocusResultResponse)
        async def run_autofocus(request: AutofocusRequest):
            """
            Run autofocus scan on specified axis.
            
            This is a synchronous operation that waits for completion.
            """
            try:
                # Run autofocus
                success = self.app_instance.autofocus.run_autofocus(
                    axis=request.axis,
                    scan_range=request.range,
                    step_size=request.step,
                    enable_plot=request.enable_plot
                )
                
                if not success:
                    raise HTTPException(status_code=500, detail="Autofocus scan failed")
                
                # Get results
                return AutofocusResultResponse(
                    axis=request.axis,
                    best_position=self.app_instance.autofocus.best_position,
                    best_metric=self.app_instance.autofocus.best_metric,
                    scan_complete=True
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Autofocus failed: {str(e)}")
        
        @app.get("/autofocus/results", response_model=AutofocusResultResponse)
        async def get_autofocus_results():
            """Get results from last autofocus scan."""
            try:
                if not self.app_instance.autofocus.positions:
                    raise HTTPException(status_code=404, detail="No autofocus results available")
                
                return AutofocusResultResponse(
                    axis=self.app_instance.autofocus.axis,
                    best_position=self.app_instance.autofocus.best_position,
                    best_metric=self.app_instance.autofocus.best_metric,
                    scan_complete=True
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")
        
        @app.post("/autofocus/save", response_model=StatusResponse)
        async def save_autofocus_results(filename: str = "autofocus_results.txt"):
            """Save last autofocus results to file."""
            try:
                self.app_instance.autofocus.save_results(filename)
                return StatusResponse(
                    status="ok",
                    message=f"Results saved to {filename}"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to save results: {str(e)}")
        
        # ========================================
        # Camera Control
        # ========================================
        
        @app.post("/camera/roi", response_model=StatusResponse)
        async def set_roi(request: ROIRequest):
            """Set camera region of interest."""
            try:
                self.app_instance.camera_app.set_roi(
                    request.left,
                    request.top,
                    request.width,
                    request.height
                )
                return StatusResponse(
                    status="ok",
                    message="ROI set successfully"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to set ROI: {str(e)}")
        
        @app.get("/camera/info")
        async def get_camera_info():
            """Get camera information and current settings."""
            try:
                info = self.app_instance.camera.get_camera_info()
                exposure = self.app_instance.camera.get_exposure_time()
                sensor_size = self.app_instance.camera.get_sensor_size()
                
                return {
                    "model": info.get("model", "Unknown"),
                    "serial": info.get("serial", "Unknown"),
                    "exposure_time": exposure,
                    "sensor_width": sensor_size[0],
                    "sensor_height": sensor_size[1]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get camera info: {str(e)}")
        
        # ========================================
        # System Control
        # ========================================
        
        @app.post("/stop", response_model=StatusResponse)
        async def stop_system():
            """Stop the experiment control system."""
            try:
                self.app_instance.stop_event.set()
                return StatusResponse(
                    status="ok",
                    message="System shutdown initiated"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to stop system: {str(e)}")
    
    def get_app(self):
        """Get the FastAPI application instance."""
        return self.fastapi_app