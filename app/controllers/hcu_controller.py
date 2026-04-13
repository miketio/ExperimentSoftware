# app/controllers/hcu_controller.py
"""
HCU Stage Controller

Wraps HCU3CStage with:
- Named preset positions (open / slit / custom)
- Persistent JSON storage in config/hcu_positions.json
- Non-blocking moves via QThread worker
- µm display, nm hardware communication
"""

from PyQt6.QtCore import QObject, QThread, pyqtSignal
import json
from pathlib import Path
from typing import Dict, Optional
import time


# ---------------------------------------------------------------------------
# Worker thread — keeps the GUI responsive during long moves
# ---------------------------------------------------------------------------

class HCUMoveWorker(QThread):
    """Execute one or more axis moves on the HCU stage in a background thread."""

    complete = pyqtSignal()
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)   # status text

    def __init__(self, hcu_stage, moves: Dict[str, int]):
        """
        Args:
            hcu_stage : HCU3CStage instance (nm units)
            moves     : {axis: pos_nm} — axes to move, in order
        """
        super().__init__()
        self.hcu_stage = hcu_stage
        self.moves = moves          # e.g. {'x': -14_000_000, 'y': 0, 'z': 0}
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            for axis, pos_nm in self.moves.items():
                if self.cancelled:
                    return
                pos_um = pos_nm / 1000.0
                self.progress.emit(
                    f"HCU: moving {axis.upper()} → {pos_um:.1f} µm …"
                )
                self.hcu_stage.move_abs(axis, pos_nm)
                time.sleep(0.05)   # small settle between axes

            self.complete.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class HCUController(QObject):
    """
    High-level controller for the HCU 3-axis stage.

    Presets
    -------
    'open'  — stage pulled away so the full image is visible (default: −14 mm on X)
    'slit'  — stage pushed in so the slit/filter is active  (default: +14 mm on X)
    Any other string key → stored in 'custom' dict

    All preset positions are saved to ``config/hcu_positions.json`` and survive
    application restarts.
    """

    POSITIONS_FILE = "config/hcu_positions.json"

    # Fallback defaults (nm).  User can override by saving from the panel.
    DEFAULT_OPEN = {'x': -14_000_000, 'y': 0, 'z': 0}
    DEFAULT_SLIT = {'x':  14_000_000, 'y': 0, 'z': 0}

    def __init__(self, state, signals, hcu_stage, parent=None):
        """
        Args:
            state     : SystemState
            signals   : SystemSignals
            hcu_stage : HCU3CStage instance  (nm)  — may be None if not connected
        """
        super().__init__(parent)
        self.state     = state
        self.signals   = signals
        self.hcu_stage = hcu_stage

        self.worker: Optional[HCUMoveWorker] = None
        self._move_busy = False

        self._data = self._load()
        print(f"[HCUController] Initialized  |  HCU stage: "
              f"{'connected' if hcu_stage else 'NOT connected'}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        path = Path(self.POSITIONS_FILE)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                    # Migrate old format if needed
                    data.setdefault('open',   self.DEFAULT_OPEN.copy())
                    data.setdefault('slit',   self.DEFAULT_SLIT.copy())
                    data.setdefault('custom', {})
                    print(f"[HCUController] Loaded positions from {path}")
                    return data
            except Exception as e:
                print(f"[HCUController] Failed to load positions: {e}")

        return {
            'open':   self.DEFAULT_OPEN.copy(),
            'slit':   self.DEFAULT_SLIT.copy(),
            'custom': {}
        }

    def _save(self):
        path = Path(self.POSITIONS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._data, f, indent=2)
        print(f"[HCUController] Positions saved to {path}")

    # ------------------------------------------------------------------
    # Position queries (µm for display)
    # ------------------------------------------------------------------

    def get_current_um(self) -> Dict[str, float]:
        """Return current HCU position in µm.  Returns zeros if not connected."""
        if self.hcu_stage is None:
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}
        try:
            return {ax: self.hcu_stage.get_pos(ax) / 1000.0 for ax in ('x', 'y', 'z')}
        except Exception as e:
            print(f"[HCUController] get_current_um error: {e}")
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}

    def get_preset_um(self, name: str) -> Optional[Dict[str, float]]:
        """Return named preset in µm, or None if not found."""
        pos_nm = self._get_preset_nm(name)
        if pos_nm is None:
            return None
        return {k: v / 1000.0 for k, v in pos_nm.items()}

    def _get_preset_nm(self, name: str) -> Optional[Dict[str, int]]:
        if name in ('open', 'slit'):
            return self._data[name]
        return self._data['custom'].get(name)

    def list_custom_presets(self) -> list:
        return list(self._data['custom'].keys())

    # ------------------------------------------------------------------
    # Saving presets
    # ------------------------------------------------------------------

    def save_current_as(self, name: str) -> bool:
        """Read current hardware position and save it under *name*."""
        if self.hcu_stage is None:
            self.signals.error_occurred.emit(
                "HCU Not Connected", "Cannot save — HCU stage not connected"
            )
            return False
        try:
            pos = {ax: int(self.hcu_stage.get_pos(ax)) for ax in ('x', 'y', 'z')}
            self._store_preset(name, pos)
            self._save()
            self.signals.status_message.emit(
                f"HCU preset '{name}' saved at "
                f"X={pos['x']/1000:.1f} Y={pos['y']/1000:.1f} Z={pos['z']/1000:.1f} µm"
            )
            return True
        except Exception as e:
            self.signals.error_occurred.emit("HCU Save Failed", str(e))
            return False

    def save_values_as(self, name: str, x_nm: int, y_nm: int, z_nm: int) -> bool:
        """Save explicit nm values under *name*."""
        self._store_preset(name, {'x': x_nm, 'y': y_nm, 'z': z_nm})
        self._save()
        self.signals.status_message.emit(
            f"HCU preset '{name}' saved "
            f"X={x_nm/1000:.1f} Y={y_nm/1000:.1f} Z={z_nm/1000:.1f} µm"
        )
        return True

    def _store_preset(self, name: str, pos_nm: dict):
        if name in ('open', 'slit'):
            self._data[name] = pos_nm
        else:
            self._data['custom'][name] = pos_nm

    def delete_custom_preset(self, name: str):
        if name in self._data['custom']:
            del self._data['custom'][name]
            self._save()

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move_away(self) -> bool:
        """Move to 'open' preset (full image visible)."""
        return self._move_to_preset('open')

    def move_in(self) -> bool:
        """Move to 'slit' preset (filter active)."""
        return self._move_to_preset('slit')

    def _move_to_preset(self, name: str) -> bool:
        pos_nm = self._get_preset_nm(name)
        if pos_nm is None:
            self.signals.error_occurred.emit("HCU Error", f"No preset '{name}' found")
            return False
        label = 'open (full image)' if name == 'open' else 'slit (filter)'
        self.signals.status_message.emit(f"HCU → {label} …")
        return self._start_move(pos_nm)

    def move_to_nm(self, x_nm: int, y_nm: int, z_nm: int) -> bool:
        """Move to explicit position (nm)."""
        return self._start_move({'x': x_nm, 'y': y_nm, 'z': z_nm})

    def _start_move(self, pos_nm: Dict[str, int]) -> bool:
        if self.hcu_stage is None:
            self.signals.error_occurred.emit(
                "HCU Not Connected", "HCU stage is not connected"
            )
            return False

        if self.worker is not None and self.worker.isRunning():
            self.signals.warning_occurred.emit(
                "HCU Busy", "HCU stage is already moving — please wait"
            )
            return False

        self.worker = HCUMoveWorker(self.hcu_stage, pos_nm)
        self.worker.complete.connect(self._on_complete)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.progress.connect(
            lambda msg: self.signals.status_message.emit(msg)
        )
        self._move_busy = True
        self.worker.start()
        self.signals.busy_started.emit("HCU Stage Move")
        return True

    def cancel_move(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.signals.status_message.emit("HCU move cancelled")

    def shutdown(self):
        """Stop any running worker so app shutdown cannot destroy an active QThread."""
        if self.worker is None:
            return

        if self.worker.isRunning():
            print("[HCUController] Waiting for active HCU move to stop...")
            self.worker.cancel()
            self.worker.wait()

        self._on_worker_finished()

    # ------------------------------------------------------------------
    # Internal signal handlers
    # ------------------------------------------------------------------

    def _on_complete(self):
        if self._move_busy:
            self.signals.busy_ended.emit()
            self._move_busy = False
        self.signals.status_message.emit("✅ HCU move complete")

    def _on_error(self, error: str):
        if self._move_busy:
            self.signals.busy_ended.emit()
            self._move_busy = False
        self.signals.error_occurred.emit("HCU Move Failed", error)

    def _on_worker_finished(self):
        """Always cleanup worker, including silent-cancel path."""
        if self._move_busy:
            self.signals.busy_ended.emit()
            self._move_busy = False

        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self.hcu_stage is not None

    @property
    def open_position_um(self) -> Dict[str, float]:
        return {k: v / 1000.0 for k, v in self._data['open'].items()}

    @property
    def slit_position_um(self) -> Dict[str, float]:
        return {k: v / 1000.0 for k, v in self._data['slit'].items()}