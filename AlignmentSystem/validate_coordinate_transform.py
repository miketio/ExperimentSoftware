#!/usr/bin/env python3
# validate_coordinate_transform.py
"""
Comprehensive validation of coordinate transform inverse operations.

Tests:
1. Round-trip accuracy (design -> stage -> design)
2. Inverse transform consistency
3. Grid-based sampling across the layout
4. Visualization of errors
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from typing import List, Tuple, Dict, Optional
import sys
from pathlib import Path

# Import the coordinate transform module
from AlignmentSystem.coordinate_transform import CoordinateTransform
from config.layout_config_generator_v2 import load_layout_config_v2


class CoordinateTransformValidator:
    """
    Validates coordinate transform inverse operations with comprehensive testing and visualization.
    """
    
    def __init__(self, transform: CoordinateTransform, layout: Optional[Dict] = None):
        """
        Initialize validator.
        
        Args:
            transform: CoordinateTransform instance to validate
            layout: Optional layout dict for context
        """
        self.transform = transform
        self.layout = layout
        self.test_results = {
            'round_trip_errors': [],
            'inverse_errors': [],
            'grid_test_points': [],
            'max_error': 0.0,
            'mean_error': 0.0,
            'passed': False
        }
    
    def validate_round_trip(self, test_points: List[Tuple[float, float]], 
                           tolerance_um: float = 0.001) -> Dict:
        """
        Test round-trip transformation: design -> stage -> design.
        
        Args:
            test_points: List of (u, v) design coordinates in µm
            tolerance_um: Maximum acceptable error in µm
        
        Returns:
            dict with test results
        """
        print("\n" + "="*70)
        print("ROUND-TRIP VALIDATION: Design -> Stage -> Design")
        print("="*70)
        
        errors = []
        max_error = 0.0
        failed_points = []
        
        for i, (u_orig, v_orig) in enumerate(test_points):
            # Forward: design -> stage
            try:
                Y_stage, Z_stage = self.transform.design_to_stage(u_orig, v_orig)
            except Exception as e:
                print(f"❌ Point {i}: Forward transform failed: {e}")
                failed_points.append((u_orig, v_orig, f"Forward: {e}"))
                continue
            
            # Inverse: stage -> design
            try:
                u_back, v_back = self.transform.stage_to_design(Y_stage, Z_stage)
            except Exception as e:
                print(f"❌ Point {i}: Inverse transform failed: {e}")
                failed_points.append((u_orig, v_orig, f"Inverse: {e}"))
                continue
            
            # Calculate error
            error_u = abs(u_back - u_orig)
            error_v = abs(v_back - v_orig)
            error_magnitude = np.hypot(error_u, error_v)
            
            errors.append({
                'point_id': i,
                'original': (u_orig, v_orig),
                'stage': (Y_stage, Z_stage),
                'recovered': (u_back, v_back),
                'error_u': error_u,
                'error_v': error_v,
                'error_magnitude': error_magnitude
            })
            
            max_error = max(max_error, error_magnitude)
            
            # Print detailed info for points with large errors
            if error_magnitude > tolerance_um:
                print(f"⚠️  Point {i}: Error = {error_magnitude:.6f} µm (exceeds tolerance)")
                print(f"   Original:  ({u_orig:.3f}, {v_orig:.3f}) µm")
                print(f"   Recovered: ({u_back:.6f}, {v_back:.6f}) µm")
                print(f"   Stage:     ({Y_stage:.1f}, {Z_stage:.1f}) nm")
        
        # Statistics
        if errors:
            error_magnitudes = [e['error_magnitude'] for e in errors]
            mean_error = np.mean(error_magnitudes)
            std_error = np.std(error_magnitudes)
            median_error = np.median(error_magnitudes)
            
            print(f"\n{'─'*70}")
            print("Round-Trip Statistics:")
            print(f"  Points tested:   {len(test_points)}")
            print(f"  Successful:      {len(errors)}")
            print(f"  Failed:          {len(failed_points)}")
            print(f"  Mean error:      {mean_error:.9f} µm ({mean_error*1000:.6f} nm)")
            print(f"  Std dev:         {std_error:.9f} µm")
            print(f"  Median error:    {median_error:.9f} µm")
            print(f"  Max error:       {max_error:.9f} µm ({max_error*1000:.6f} nm)")
            print(f"  Tolerance:       {tolerance_um:.9f} µm ({tolerance_um*1000:.6f} nm)")
            
            passed = max_error <= tolerance_um and len(failed_points) == 0
            
            if passed:
                print(f"\n✅ PASSED: All points within tolerance")
            else:
                print(f"\n❌ FAILED: {len([e for e in errors if e['error_magnitude'] > tolerance_um])} points exceed tolerance")
            
            return {
                'passed': passed,
                'errors': errors,
                'failed_points': failed_points,
                'mean_error': mean_error,
                'max_error': max_error,
                'std_error': std_error,
                'median_error': median_error,
                'tolerance': tolerance_um
            }
        else:
            print(f"\n❌ FAILED: No successful transformations")
            return {
                'passed': False,
                'errors': [],
                'failed_points': failed_points,
                'mean_error': float('inf'),
                'max_error': float('inf'),
                'tolerance': tolerance_um
            }
    
    def validate_grid(self, u_range: Tuple[float, float], v_range: Tuple[float, float],
                     grid_points: int = 20, tolerance_um: float = 0.001) -> Dict:
        """
        Test round-trip on a regular grid across the layout.
        
        Args:
            u_range: (u_min, u_max) in µm
            v_range: (v_min, v_max) in µm
            grid_points: Number of points per dimension
            tolerance_um: Maximum acceptable error in µm
        
        Returns:
            dict with test results
        """
        print("\n" + "="*70)
        print(f"GRID VALIDATION: {grid_points}x{grid_points} points")
        print("="*70)
        
        # Generate grid
        u_vals = np.linspace(u_range[0], u_range[1], grid_points)
        v_vals = np.linspace(v_range[0], v_range[1], grid_points)
        
        test_points = []
        for u in u_vals:
            for v in v_vals:
                test_points.append((u, v))
        
        print(f"Testing {len(test_points)} grid points...")
        print(f"  U range: [{u_range[0]:.1f}, {u_range[1]:.1f}] µm")
        print(f"  V range: [{v_range[0]:.1f}, {v_range[1]:.1f}] µm")
        
        # Run validation
        result = self.validate_round_trip(test_points, tolerance_um)
        
        # Store grid info for plotting
        result['grid_shape'] = (grid_points, grid_points)
        result['u_range'] = u_range
        result['v_range'] = v_range
        
        return result
    
    def validate_specific_features(self) -> Dict:
        """
        Validate round-trip for specific layout features (fiducials, gratings).
        Requires layout to be set.
        
        Returns:
            dict with test results
        """
        if self.layout is None:
            print("⚠️  No layout provided - skipping feature validation")
            return {'passed': False, 'errors': [], 'message': 'No layout'}
        
        print("\n" + "="*70)
        print("FEATURE VALIDATION: Fiducials and Gratings")
        print("="*70)
        
        test_points = []
        point_labels = []
        
        # Extract fiducial positions from all blocks
        for block_id, block in self.layout['blocks'].items():
            block_center = block['design_position']
            block_size = self.layout['block_layout']['block_size']
            
            # Convert block center to bottom-left
            u_bl = block_center[0] - block_size / 2.0
            v_bl = block_center[1] - block_size / 2.0
            
            # Test all fiducials
            for corner, local_pos in block['fiducials'].items():
                u_global = u_bl + local_pos[0]
                v_global = v_bl + local_pos[1]
                test_points.append((u_global, v_global))
                point_labels.append(f"Block{block_id}_{corner}")
            
            # Test gratings (if any)
            for grating_id, grating in block.get('gratings', {}).items():
                u_local, v_local = grating['position']
                u_global = u_bl + u_local
                v_global = v_bl + v_local
                test_points.append((u_global, v_global))
                point_labels.append(f"Block{block_id}_{grating_id}")
        
        print(f"Testing {len(test_points)} layout features...")
        
        # Run validation
        result = self.validate_round_trip(test_points, tolerance_um=0.001)
        result['point_labels'] = point_labels
        
        return result
    
    def plot_validation_results(self, result: Dict, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of validation results.
        
        Args:
            result: Dict from validate_round_trip or validate_grid
            save_path: Optional path to save figure
        """
        if not result.get('errors'):
            print("⚠️  No results to plot")
            return
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        errors = result['errors']
        error_magnitudes = np.array([e['error_magnitude'] for e in errors])
        original_points = np.array([e['original'] for e in errors])
        recovered_points = np.array([e['recovered'] for e in errors])
        
        # =====================================================================
        # Plot 1: Error Magnitude Heatmap (if grid)
        # =====================================================================
        ax1 = fig.add_subplot(gs[0, 0])
        
        if 'grid_shape' in result:
            grid_h, grid_w = result['grid_shape']
            error_grid = error_magnitudes.reshape(grid_h, grid_w)
            
            im1 = ax1.imshow(error_grid, cmap='hot', origin='lower', 
                            extent=[result['u_range'][0], result['u_range'][1],
                                   result['v_range'][0], result['v_range'][1]],
                            aspect='auto')
            ax1.set_title('Round-Trip Error Magnitude (µm)', fontweight='bold')
            ax1.set_xlabel('U (µm)')
            ax1.set_ylabel('V (µm)')
            plt.colorbar(im1, ax=ax1, label='Error (µm)')
        else:
            # Scatter plot if not grid
            scatter = ax1.scatter(original_points[:, 0], original_points[:, 1],
                                 c=error_magnitudes, cmap='hot', s=50, alpha=0.7)
            ax1.set_title('Round-Trip Error by Position', fontweight='bold')
            ax1.set_xlabel('U (µm)')
            ax1.set_ylabel('V (µm)')
            ax1.set_aspect('equal')
            plt.colorbar(scatter, ax=ax1, label='Error (µm)')
        
        # =====================================================================
        # Plot 2: Error Histogram
        # =====================================================================
        ax2 = fig.add_subplot(gs[0, 1])
        
        ax2.hist(error_magnitudes * 1000, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(result['mean_error'] * 1000, color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {result['mean_error']*1000:.3f} nm")
        ax2.axvline(result['max_error'] * 1000, color='orange', linestyle='--',
                   linewidth=2, label=f"Max: {result['max_error']*1000:.3f} nm")
        if result.get('tolerance'):
            ax2.axvline(result['tolerance'] * 1000, color='green', linestyle=':',
                       linewidth=2, label=f"Tolerance: {result['tolerance']*1000:.3f} nm")
        
        ax2.set_xlabel('Error (nm)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 3: Error Statistics Box
        # =====================================================================
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        stats_text = f"""
ROUND-TRIP VALIDATION SUMMARY
{'='*35}

Points Tested:     {len(errors)}
Failed Points:     {len(result.get('failed_points', []))}

ERROR STATISTICS (µm):
  Mean:      {result['mean_error']:.9f}
  Median:    {result.get('median_error', 0):.9f}
  Std Dev:   {result.get('std_error', 0):.9f}
  Max:       {result['max_error']:.9f}
  
ERROR STATISTICS (nm):
  Mean:      {result['mean_error']*1000:.6f}
  Median:    {result.get('median_error', 0)*1000:.6f}
  Max:       {result['max_error']*1000:.6f}

TOLERANCE:         {result.get('tolerance', 0):.9f} µm
                   ({result.get('tolerance', 0)*1000:.6f} nm)

STATUS: {'✅ PASSED' if result['passed'] else '❌ FAILED'}
"""
        
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
                fontsize=9, family='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # =====================================================================
        # Plot 4: U-component errors
        # =====================================================================
        ax4 = fig.add_subplot(gs[1, 0])
        
        error_u = np.array([e['error_u'] for e in errors]) * 1000  # to nm
        ax4.scatter(original_points[:, 0], error_u, alpha=0.5, s=20)
        ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax4.axhline(np.mean(error_u), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(error_u):.3f} nm')
        ax4.set_xlabel('U (µm)')
        ax4.set_ylabel('U Error (nm)')
        ax4.set_title('U-Component Error vs Position', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 5: V-component errors
        # =====================================================================
        ax5 = fig.add_subplot(gs[1, 1])
        
        error_v = np.array([e['error_v'] for e in errors]) * 1000  # to nm
        ax5.scatter(original_points[:, 1], error_v, alpha=0.5, s=20)
        ax5.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax5.axhline(np.mean(error_v), color='red', linestyle='--',
                   label=f'Mean: {np.mean(error_v):.3f} nm')
        ax5.set_xlabel('V (µm)')
        ax5.set_ylabel('V Error (nm)')
        ax5.set_title('V-Component Error vs Position', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 6: Vector field of errors (magnified)
        # =====================================================================
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Sample subset for clarity
        step = max(1, len(errors) // 100)
        sample_orig = original_points[::step]
        sample_recov = recovered_points[::step]
        
        # Error vectors (magnified for visibility)
        magnification = 1e6  # 1 µm error = 1000000 display units
        du = (sample_recov[:, 0] - sample_orig[:, 0]) * magnification
        dv = (sample_recov[:, 1] - sample_orig[:, 1]) * magnification
        
        ax6.quiver(sample_orig[:, 0], sample_orig[:, 1], du, dv,
                  error_magnitudes[::step], cmap='hot', alpha=0.6,
                  scale=1, scale_units='xy', angles='xy', width=0.003)
        ax6.set_xlabel('U (µm)')
        ax6.set_ylabel('V (µm)')
        ax6.set_title('Error Vectors (Magnified 1e6×)', fontweight='bold')
        ax6.set_aspect('equal')
        ax6.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 7: Error vs Distance from Origin
        # =====================================================================
        ax7 = fig.add_subplot(gs[2, 0])
        
        distances = np.hypot(original_points[:, 0], original_points[:, 1])
        ax7.scatter(distances, error_magnitudes * 1000, alpha=0.5, s=20)
        ax7.set_xlabel('Distance from Origin (µm)')
        ax7.set_ylabel('Error (nm)')
        ax7.set_title('Error vs Distance from Origin', fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 8: Original vs Recovered (overlay)
        # =====================================================================
        ax8 = fig.add_subplot(gs[2, 1])
        
        ax8.scatter(original_points[:, 0], original_points[:, 1], 
                   c='blue', marker='o', s=30, alpha=0.5, label='Original')
        ax8.scatter(recovered_points[:, 0], recovered_points[:, 1],
                   c='red', marker='x', s=30, alpha=0.5, label='Recovered')
        
        # Draw error vectors for worst cases
        worst_indices = np.argsort(error_magnitudes)[-10:]  # 10 worst
        for idx in worst_indices:
            ax8.plot([original_points[idx, 0], recovered_points[idx, 0]],
                    [original_points[idx, 1], recovered_points[idx, 1]],
                    'orange', linewidth=1, alpha=0.5)
        
        ax8.set_xlabel('U (µm)')
        ax8.set_ylabel('V (µm)')
        ax8.set_title('Original vs Recovered Positions', fontweight='bold')
        ax8.legend()
        ax8.set_aspect('equal')
        ax8.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 9: Cumulative Error Distribution
        # =====================================================================
        ax9 = fig.add_subplot(gs[2, 2])
        
        sorted_errors = np.sort(error_magnitudes) * 1000  # nm
        percentiles = np.linspace(0, 100, len(sorted_errors))
        
        ax9.plot(percentiles, sorted_errors, linewidth=2)
        ax9.axhline(result['mean_error'] * 1000, color='red', linestyle='--',
                   label=f"Mean: {result['mean_error']*1000:.3f} nm")
        ax9.axhline(np.percentile(error_magnitudes * 1000, 95), 
                   color='orange', linestyle='--',
                   label=f"95th %ile: {np.percentile(error_magnitudes * 1000, 95):.3f} nm")
        
        ax9.set_xlabel('Percentile')
        ax9.set_ylabel('Error (nm)')
        ax9.set_title('Cumulative Error Distribution', fontweight='bold')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # =====================================================================
        # Overall title
        # =====================================================================
        status_color = 'green' if result['passed'] else 'red'
        status_text = '✅ PASSED' if result['passed'] else '❌ FAILED'
        
        fig.suptitle(f'Coordinate Transform Round-Trip Validation - {status_text}',
                    fontsize=16, fontweight='bold', color=status_color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"\n💾 Saved validation plot: {save_path}")
        
        plt.show()


def run_comprehensive_validation():
    """
    Run complete validation suite on coordinate transform.
    
    NOTE: This will expose a bug in coordinate_transform.py if present!
    The calibrate() method should pass design_nm (not design) to 
    _calibrate_two_points and _calibrate_least_squares.
    """
    print("\n" + "="*70)
    print("COORDINATE TRANSFORM COMPREHENSIVE VALIDATION")
    print("="*70)
    print("\n⚠️  WARNING: This test will expose coordinate system bugs!")
    print("   If you see large errors (meters instead of nanometers),")
    print("   check the calibrate() method in coordinate_transform.py\n")
    
    # Load layout and setup transform
    layout_path = "config/mock_layout.json"
    print(f"\nLoading layout: {layout_path}")
    
    try:
        layout = load_layout_config_v2(layout_path)
    except Exception as e:
        print(f"❌ Failed to load layout: {e}")
        return 1
    
    # Create transform with ground truth
    transform = CoordinateTransform(layout)
    gt = layout['simulation_ground_truth']
    transform.set_transformation(gt['rotation_deg'], tuple(gt['translation_um']))
    
    print(f"✅ Transform initialized")
    print(f"   Rotation: {gt['rotation_deg']}°")
    print(f"   Translation: {gt['translation_um']} um")
    
    # Create validator
    validator = CoordinateTransformValidator(transform, layout)
    
    # Test 1: Grid validation (comprehensive)
    print("\n" + "="*70)
    print("TEST 1: GRID VALIDATION")
    print("="*70)
    
    grid_result = validator.validate_grid(
        u_range=(0, 1400),  # Full layout width
        v_range=(-900, 0),  # Full layout height
        grid_points=25,
        tolerance_um=0.001
    )
    
    validator.plot_validation_results(grid_result, 
                                     save_path='validation_grid_roundtrip.png')
    
    # Test 2: Feature validation (fiducials and gratings)
    print("\n" + "="*70)
    print("TEST 2: FEATURE VALIDATION")
    print("="*70)
    
    feature_result = validator.validate_specific_features()
    
    if feature_result.get('errors'):
        validator.plot_validation_results(feature_result,
                                         save_path='validation_features_roundtrip.png')
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_passed = grid_result['passed'] and feature_result.get('passed', True)
    
    print(f"\nGrid Test:     {'✅ PASSED' if grid_result['passed'] else '❌ FAILED'}")
    print(f"  Max error:   {grid_result['max_error']*1000:.6f} nm")
    print(f"  Mean error:  {grid_result['mean_error']*1000:.6f} nm")
    
    if feature_result.get('errors'):
        print(f"\nFeature Test:  {'✅ PASSED' if feature_result['passed'] else '❌ FAILED'}")
        print(f"  Max error:   {feature_result['max_error']*1000:.6f} nm")
        print(f"  Mean error:  {feature_result['mean_error']*1000:.6f} nm")
    
    print(f"\n{'='*70}")
    if all_passed:
        print("🎉 ALL TESTS PASSED - Coordinate transform is accurate!")
    else:
        print("⚠️  SOME TESTS FAILED - Review results above")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_comprehensive_validation())