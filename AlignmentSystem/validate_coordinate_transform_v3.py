#!/usr/bin/env python3
# validate_coordinate_transform_v3.py
"""
Validation suite for CoordinateTransformV3 (clean, model-consistent).

Features:
  - Uses CameraLayout.from_json_file()
  - Computes design bounds from block geometry
  - Performs grid and feature round-trip tests
  - Produces comprehensive 2×2 validation plot:
      (overlay, vector field, histogram, stats)
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

from AlignmentSystem.coordinate_transform_v3 import CoordinateTransformV3
from config.layout_models import CameraLayout, Block


# ============================================================
# Utilities for computing design-space bounds
# ============================================================
def block_bottom_left_global(block: Block, block_size: float) -> Tuple[float, float]:
    """Return bottom-left corner (global design coords) from block center."""
    u_center, v_center = block.design_position.u, block.design_position.v
    return u_center - block_size / 2.0, v_center - block_size / 2.0


def compute_design_bounds(layout: CameraLayout) -> Tuple[float, float, float, float]:
    """Compute (umin, umax, vmin, vmax) from all features in the layout."""
    xs, ys = [], []
    block_size = layout.block_layout.block_size

    for block in layout.blocks.values():
        u_bl, v_bl = block_bottom_left_global(block, block_size)

        # block corners
        xs += [u_bl, u_bl + block_size]
        ys += [v_bl, v_bl + block_size]

        # fiducials
        for fid in block.fiducials.values():
            xs.append(u_bl + fid.u)
            ys.append(v_bl + fid.v)

        # gratings
        for gr in block.gratings.values():
            xs.append(u_bl + gr.position.u)
            ys.append(v_bl + gr.position.v)

        # waveguides
        for wg in block.waveguides.values():
            xs += [u_bl + wg.u_start, u_bl + wg.u_end, u_bl + wg.center_position.u]
            ys += [v_bl + wg.v_center, v_bl + wg.v_center, v_bl + wg.center_position.v]

    umin, umax = min(xs), max(xs)
    vmin, vmax = min(ys), max(ys)

    margin_u = 0.01 * (umax - umin)
    margin_v = 0.01 * (vmax - vmin)
    return umin - margin_u, umax + margin_u, vmin - margin_v, vmax + margin_v


# ============================================================
# Validation class
# ============================================================
class CoordinateTransformValidator:
    """Validator for CoordinateTransformV3 using CameraLayout geometry."""

    def __init__(self, transform: CoordinateTransformV3, layout: Optional[CameraLayout] = None):
        self.transform = transform
        self.layout = layout

    # --------------------------
    # Core round-trip validator
    # --------------------------
    def validate_round_trip(self,
                            points: List[Tuple[float, float]],
                            tolerance_um: float = 1e-3) -> Dict:
        detailed = []
        failed = []

        for (u, v) in points:
            try:
                Y, Z = self.transform.design_to_stage(u, v)
                u_back, v_back = self.transform.stage_to_design(Y, Z)
            except Exception as e:
                failed.append((u, v, str(e)))
                continue

            du = float(u_back - u)
            dv = float(v_back - v)
            err = float(np.hypot(du, dv))

            detailed.append({
                "original": (float(u), float(v)),
                "stage": (float(Y), float(Z)),
                "recovered": (float(u_back), float(v_back)),
                "error_u": du,
                "error_v": dv,
                "error_magnitude": err,
            })

        if not detailed:
            return {
                "passed": False,
                "errors": [],
                "failed": failed,
                "mean_error": float("inf"),
                "max_error": float("inf"),
                "std_error": float("inf"),
                "median_error": float("inf"),
                "tolerance": tolerance_um,
                "num_tested": len(points),
            }

        error_mags = np.array([d["error_magnitude"] for d in detailed])
        stats = {
            "mean_error": float(np.mean(error_mags)),
            "max_error": float(np.max(error_mags)),
            "std_error": float(np.std(error_mags)),
            "median_error": float(np.median(error_mags)),
        }
        passed = stats["max_error"] <= tolerance_um and not failed

        return {
            "passed": passed,
            "errors": detailed,
            "failed": failed,
            **stats,
            "tolerance": tolerance_um,
            "num_tested": len(points),
        }

    # --------------------------
    # Grid validation
    # --------------------------
    def validate_grid(self,
                      u_range: Tuple[float, float],
                      v_range: Tuple[float, float],
                      grid_points: int = 20,
                      tolerance_um: float = 1e-3) -> Dict:
        u_vals = np.linspace(u_range[0], u_range[1], grid_points)
        v_vals = np.linspace(v_range[0], v_range[1], grid_points)
        pts = [(u, v) for u in u_vals for v in v_vals]
        print(f"Running grid validation with {len(pts)} points...")
        return self.validate_round_trip(pts, tolerance_um)

    # --------------------------
    # Feature validation
    # --------------------------
    def validate_features(self, tolerance_um: float = 1e-3) -> Dict:
        if self.layout is None:
            print("No layout provided, skipping feature validation.")
            return {"passed": False, "errors": []}

        pts = []
        block_size = self.layout.block_layout.block_size

        for block in self.layout.blocks.values():
            u_bl, v_bl = block_bottom_left_global(block, block_size)
            for fid in block.fiducials.values():
                pts.append((u_bl + fid.u, v_bl + fid.v))
            for gr in block.gratings.values():
                pts.append((u_bl + gr.position.u, v_bl + gr.position.v))
            for wg in block.waveguides.values():
                pts.append((u_bl + wg.u_start, v_bl + wg.v_center))
                pts.append((u_bl + wg.u_end, v_bl + wg.v_center))

        print(f"Running feature validation on {len(pts)} points...")
        return self.validate_round_trip(pts, tolerance_um)

    # --------------------------
    # Plot full validation report
    # --------------------------
    def plot_validation_results(self, result: Dict, save_path: Optional[str] = None):
        if not result.get("errors"):
            print("No results to plot.")
            return

        errors = result["errors"]
        error_magnitudes = np.array([e["error_magnitude"] for e in errors])
        orig = np.array([e["original"] for e in errors])
        recov = np.array([e["recovered"] for e in errors])

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Overlay
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(orig[:, 0], orig[:, 1], c="blue", s=40, alpha=0.6, label="Original")
        ax1.scatter(recov[:, 0], recov[:, 1], c="red", marker="x", s=40, alpha=0.6, label="Recovered")
        worst = np.argsort(error_magnitudes)[-15:]
        for i in worst:
            ax1.plot([orig[i, 0], recov[i, 0]], [orig[i, 1], recov[i, 1]], "orange", alpha=0.7)
        ax1.legend()
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Original vs Recovered")

        # Vector field
        ax2 = fig.add_subplot(gs[0, 1])
        step = max(1, len(errors) // 200)
        du = (recov[:, 0] - orig[:, 0])[::step]
        dv = (recov[:, 1] - orig[:, 1])[::step]
        sample = orig[::step]
        q = ax2.quiver(sample[:, 0], sample[:, 1], du * 1e3, dv * 1e3, error_magnitudes[::step],
                       cmap="hot", scale=1, scale_units="xy", angles="xy")
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Error Vector Field (×1000 for visibility)")
        plt.colorbar(q, ax=ax2, label="Error (µm)")

        # Histogram
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(error_magnitudes, bins=50, edgecolor="black", alpha=0.7)
        ax3.axvline(result["mean_error"], color="red", linestyle="--", label=f"Mean {result['mean_error']:.6f}")
        ax3.axvline(result["max_error"], color="orange", linestyle="--", label=f"Max {result['max_error']:.6f}")
        if result.get("tolerance"):
            ax3.axvline(result["tolerance"], color="green", linestyle=":", label=f"Tolerance {result['tolerance']:.6f}")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Error Distribution")

        # Stats summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        p50, p95, p99 = np.percentile(error_magnitudes, [50, 95, 99])
        text = f"""
ROUND-TRIP VALIDATION SUMMARY
{'='*40}
Points tested: {result['num_tested']}
Failed points: {len(result['failed'])}

Mean error:   {result['mean_error']:.9f} µm
Median:       {result['median_error']:.9f} µm
Std:          {result['std_error']:.9f} µm
Max:          {result['max_error']:.9f} µm

50th: {p50:.9f} µm
95th: {p95:.9f} µm
99th: {p99:.9f} µm

Status: {'PASSED' if result['passed'] else 'FAILED'}
"""
        ax4.text(0.05, 0.95, text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment="top", family="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightgreen" if result["passed"] else "lightsalmon", alpha=0.8))

        plt.suptitle("Coordinate Transform Round-Trip Validation", fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved validation plot -> {save_path}")

        plt.show()


# ============================================================
# CLI entry point
# ============================================================
def run_validation(layout_path: str = "config/mock_layout.json") -> int:
    p = Path(layout_path)
    if not p.exists():
        print(f"Layout file not found: {layout_path}")
        return 2

    layout = CameraLayout.from_json_file(layout_path)
    print(f"Loaded CameraLayout '{layout.design_name}' with {len(layout.blocks)} blocks")

    transform = CoordinateTransformV3(layout)
    gt = layout.ground_truth
    transform.set_transformation(gt.rotation_deg, tuple(gt.translation_um.to_tuple()))
    print(f"Applied ground-truth transform: rotation={gt.rotation_deg}°, translation={gt.translation_um.to_tuple()} µm")

    validator = CoordinateTransformValidator(transform, layout)
    umin, umax, vmin, vmax = compute_design_bounds(layout)
    print(f"Design bounds: u[{umin:.3f},{umax:.3f}], v[{vmin:.3f},{vmax:.3f}] µm")

    grid_result = validator.validate_grid((umin, umax), (vmin, vmax), grid_points=20)
    feature_result = validator.validate_features()

    validator.plot_validation_results(grid_result, save_path="validation_grid_full.png")
    validator.plot_validation_results(feature_result, save_path="validation_features_full.png")

    all_passed = grid_result["passed"] and feature_result["passed"]
    print("\nOverall result:", "✅ PASSED" if all_passed else "❌ FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_validation())
