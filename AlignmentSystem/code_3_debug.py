#!/usr/bin/env python3
# step3_debug.py
"""
Isolated debug script for Step 3 - Camera simulation and fiducial detection.
Run this to debug camera rendering and CV detection issues.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from AlignmentSystem.cv_tools import VisionTools
from AlignmentSystem.camera_simulation import Camera
from AlignmentSystem.ascii_parser import ASCIIParser
from AlignmentSystem.layout_config_generator import generate_layout_config


class Step3Debugger:
    """Isolated debugger for Step 3 - Camera and CV detection."""
    
    def __init__(self, ascii_file: str = "./AlignmentSystem/ascii_sample.ASC"):
        self.ascii_file = ascii_file
        self.parsed_data = None
        self.layout_config = None
        self.rotation_angle = 0.0  # Simulated rotation in degrees
        self.translation = (0, 0)  # Simulated translation (Y, Z) in nm
        self.camera = Camera(pixel_width=2048, pixel_height=2048, nm_per_pixel=1000)
        self.vt = VisionTools()
        
    def design_to_stage(self, u_um, v_um):
        """
        Convert design coordinates (u, v) in µm to stage coordinates (Y, Z) in nm.
        Applies rotation and translation.
        """
        # Convert to nm
        u_nm = u_um * 1000
        v_nm = v_um * 1000
        
        # Apply rotation around origin
        angle_rad = np.radians(self.rotation_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        Y = cos_a * u_nm - sin_a * v_nm + self.translation[0]
        Z = sin_a * u_nm + cos_a * v_nm + self.translation[1]
        
        return (Y, Z)
    
    def setup(self):
        """Parse ASCII and generate layout config."""
        print("\n" + "="*70)
        print("SETUP: Loading ASCII and generating layout")
        print("="*70)
        
        # Parse ASCII
        if not Path(self.ascii_file).exists():
            print(f"❌ ASCII file not found: {self.ascii_file}")
            return False
        
        parser = ASCIIParser(self.ascii_file)
        self.parsed_data = parser.parse()
        
        print(f"✅ Parsed: {self.ascii_file}")
        print(f"   Markers: {len(self.parsed_data['markers'])}")
        print(f"   Waveguides: {len(self.parsed_data['waveguides'])}")
        print(f"   Gratings: {len(self.parsed_data['gratings'])}")
        
        # Generate layout config
        config_file = "config/debug_layout.json"
        Path("config").mkdir(exist_ok=True)
        self.layout_config = generate_layout_config(self.ascii_file, config_file)
        
        print(f"✅ Generated layout config")
        print(f"   Total blocks: {self.layout_config['total_blocks']}")
        print(f"   Block spacing: {self.layout_config['block_spacing']} µm")
        
        return True
    
    def verify_coordinate_system(self):
        """Verify that coordinate transformations are working correctly."""
        print("\n" + "="*70)
        print("VERIFICATION: Coordinate System Check")
        print("="*70)
        
        # Test point: Block 1 top-left corner
        block1 = self.layout_config['blocks'][1]
        tl_design = block1['fiducials']['top_left']
        
        print(f"\n📍 Block 1 Top-Left Fiducial:")
        print(f"   Design coords: ({tl_design[0]:.3f}, {tl_design[1]:.3f}) µm")
        
        # Apply transformation
        tl_stage = self.design_to_stage(*tl_design)
        print(f"   Stage coords (nm): ({tl_stage[0]:.1f}, {tl_stage[1]:.1f})")
        print(f"   Stage coords (µm): ({tl_stage[0]/1000:.3f}, {tl_stage[1]/1000:.3f})")
        print(f"   Rotation applied: {self.rotation_angle}°")
        print(f"   Translation applied: {self.translation} nm")
        
        # Check what's in layout_elements
        print(f"\n🔍 Checking layout_elements generation...")
        layout_elements = self._build_layout_elements_for_camera()
        
        # Find matching fiducials for Block 1
        block1_fiducials = [
            elem for elem in layout_elements 
            if elem['type'] == 'fiducial' and elem.get('block_id') == 1
        ]
        
        print(f"\n   Block 1 fiducials in layout_elements: {len(block1_fiducials)}")
        
        tl_found = False
        for fid in block1_fiducials:
            Y_um, Z_um = fid['coords_stage']
            Y_nm = Y_um * 1000
            Z_nm = Z_um * 1000
            
            # Check if this matches our expected top-left
            if fid['corner'] == 'top_left':
                tl_found = True
                error_Y = abs(Y_nm - tl_stage[0])
                error_Z = abs(Z_nm - tl_stage[1])
                
                print(f"\n   ✓ Found top_left fiducial:")
                print(f"      coords_stage: ({Y_um:.3f}, {Z_um:.3f}) µm")
                print(f"      coords_stage: ({Y_nm:.1f}, {Z_nm:.1f}) nm")
                print(f"      Error from expected: ({error_Y:.1f}, {error_Z:.1f}) nm")
                
                if error_Y < 1 and error_Z < 1:
                    print(f"      ✅ MATCH! Coordinates are correct.")
                else:
                    print(f"      ⚠️  Mismatch detected!")
        
        if not tl_found:
            print(f"   ❌ Top-left fiducial NOT found in layout_elements!")
            print(f"   Available corners:", [f['corner'] for f in block1_fiducials])
        
        return tl_found
    
    def _build_layout_elements_for_camera(self):
        """Build layout_elements list for camera rendering."""
        layout_elements = []
        
        for block_id, block in self.layout_config['blocks'].items():
            block_center_design = block['center']  # (u, v) in µm
            
            # Add fiducials (corner markers)
            for corner_name, corner_design in block['fiducials'].items():
                # Convert to stage coordinates (returns nm)
                corner_stage_nm = self.design_to_stage(*corner_design)
                
                # Store as µm for camera
                corner_stage_um = (corner_stage_nm[0] / 1000.0, corner_stage_nm[1] / 1000.0)
                
                layout_elements.append({
                    'type': 'fiducial',
                    'coords_stage': corner_stage_um,  # (Y, Z) in µm
                    'block_id': block_id,
                    'corner': corner_name
                })
            
            # Add block outlines
            corners_design = [
                block['fiducials']['top_left'],
                block['fiducials']['top_right'],
                block['fiducials']['bottom_right'],
                block['fiducials']['bottom_left']
            ]
            
            corners_stage_um = []
            for u, v in corners_design:
                corner_stage_nm = self.design_to_stage(u, v)
                corners_stage_um.append((corner_stage_nm[0] / 1000.0, corner_stage_nm[1] / 1000.0))
            
            layout_elements.append({
                'type': 'block_outline',
                'coords_stage': corners_stage_um,
                'block_id': block_id
            })
            
            # Add waveguides from parsed_data
            # These are in block-local coordinates, need to offset by block center
            if self.parsed_data and 'waveguides' in self.parsed_data:
                for wg in self.parsed_data['waveguides']:
                    # Waveguide corners in block-local coordinates (µm)
                    u_start = wg['u_start']
                    u_end = wg['u_end']
                    v_bottom = wg['v_bottom']
                    v_top = wg['v_top']
                    
                    # Convert to global design coordinates by adding block center
                    ref_u, ref_v = block['fiducials']['bottom_left']
                    wg_corners_design = [
                        (ref_u + u_start, ref_v + v_bottom),
                        (ref_u + u_end, ref_v + v_bottom),
                        (ref_u + u_end, ref_v + v_top),
                        (ref_u + u_start, ref_v + v_top)
                    ]
                    
                    # Convert to stage coordinates (µm)
                    wg_corners_stage_um = []
                    for u, v in wg_corners_design:
                        corner_stage_nm = self.design_to_stage(u, v)
                        wg_corners_stage_um.append((corner_stage_nm[0] / 1000.0, corner_stage_nm[1] / 1000.0))
                    
                    layout_elements.append({
                        'type': 'waveguide',
                        'coords_stage': wg_corners_stage_um,  # List of (Y, Z) in µm
                        'block_id': block_id,
                        'wg_number': wg['number']
                    })
            
            # Use bottom-left fiducial as block anchor
            ref_u, ref_v = block['fiducials']['bottom_left']
            
            for name, local_coords in block['gratings'].items():
                u_local, v_local = local_coords

                # Compute global design coordinates
                u_global = ref_u + u_local
                v_global = ref_v + v_local

                # Convert to stage coordinates
                stage_nm = self.design_to_stage(u_global, v_global)
                stage_um = (stage_nm[0] / 1000.0, stage_nm[1] / 1000.0)

                side = 'left' if 'left' in name.lower() else 'right'

                layout_elements.append({
                    'type': 'grating',
                    'coords_stage': stage_um,
                    'side': side,
                    'block_id': block_id,
                    'grating_name': name
                })
        
        return layout_elements
    
    def test_camera_rendering(self):
        """Test camera rendering with detailed diagnostics."""
        print("\n" + "="*70)
        print("TEST: Camera Rendering")
        print("="*70)
        
        # Get Block 1 top-left fiducial
        block1 = self.layout_config['blocks'][1]
        tl_design = block1['fiducials']['top_left']
        tl_stage = self.design_to_stage(*tl_design)
        
        print(f"\n📷 Camera configuration:")
        print(f"   Resolution: {self.camera.pixel_width}x{self.camera.pixel_height} pixels")
        print(f"   Scale: {self.camera.nm_per_pixel} nm/pixel")
        print(f"   FOV: {self.camera.fov_width_nm/1000:.1f} x {self.camera.fov_height_nm/1000:.1f} µm")
        
        print(f"\n🎯 Target: Block 1 Top-Left Fiducial")
        print(f"   Design: ({tl_design[0]:.1f}, {tl_design[1]:.1f}) µm")
        print(f"   Stage: ({tl_stage[0]/1000:.1f}, {tl_stage[1]/1000:.1f}) µm")
        
        # Move camera to target
        print(f"\n   Moving camera to target position...")
        self.camera.move_to(*tl_stage)
        
        # Get FOV bounds
        Y_min, Y_max, Z_min, Z_max = self.camera.get_fov_bounds()
        print(f"   Camera FOV bounds:")
        print(f"      Y: [{Y_min/1000:.1f}, {Y_max/1000:.1f}] µm")
        print(f"      Z: [{Z_min/1000:.1f}, {Z_max/1000:.1f}] µm")
        
        # Build layout elements
        layout_elements = self._build_layout_elements_for_camera()
        print(f"\n   Total layout elements: {len(layout_elements)}")
        
        # Count by type
        type_counts = {}
        for elem in layout_elements:
            t = elem['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print(f"   Element breakdown:")
        for elem_type, count in type_counts.items():
            print(f"      {elem_type}: {count}")
        
        # Check which fiducials are in FOV
        print(f"\n   🔍 Checking fiducials in FOV:")
        fiducials_in_fov = []

        print("\n📊 Checking grating placement:")
        visible_count = 0
        for e in layout_elements:
            if e['type'] == 'grating':
                y, z = e['coords_stage']



                in_fov = (-1020 < y < 1020) and (-830 < z < 1220)

                print(
                    f"   Block {e['block_id']:2d} {e['grating_name']:10s}"
                    f" at ({y:8.2f}, {z:8.2f}) µm"
                    f" {'✅' if in_fov else '❌'}"
                )

                if in_fov:
                    visible_count += 1

        print(f"   → {visible_count} gratings are within FOV\n")

        for elem in layout_elements:
            if elem['type'] == 'fiducial':
                Y_um, Z_um = elem['coords_stage']
                Y_nm = Y_um * 1000
                Z_nm = Z_um * 1000
                
                if Y_min <= Y_nm <= Y_max and Z_min <= Z_nm <= Z_max:
                    dist_to_center = np.sqrt((Y_nm - tl_stage[0])**2 + (Z_nm - tl_stage[1])**2)
                    fiducials_in_fov.append({
                        'block_id': elem['block_id'],
                        'corner': elem['corner'],
                        'position': (Y_um, Z_um),
                        'distance': dist_to_center
                    })
        
        print(f"      Found {len(fiducials_in_fov)} fiducials in FOV:")
        for fid in sorted(fiducials_in_fov, key=lambda x: x['distance'])[:5]:
            print(f"         Block {fid['block_id']:2d} {fid['corner']:12s}: "
                  f"({fid['position'][0]:7.1f}, {fid['position'][1]:7.1f}) µm, "
                  f"dist={fid['distance']/1000:6.1f} µm")
        
        # Take image
        print(f"\n   📸 Taking image...")
        img = self.camera.take_image(layout_elements, debug=True)
        
        # Analyze image
        print(f"\n   📊 Image analysis:")
        print(f"      Shape: {img.shape}")
        print(f"      Dtype: {img.dtype}")
        print(f"      Range: [{img.min()}, {img.max()}]")
        print(f"      Mean: {img.mean():.1f}")
        print(f"      Non-zero pixels: {np.count_nonzero(img > 200)}")
        print(f"      Bright pixels (>1000): {np.count_nonzero(img > 1000)}")
        print(f"      Very bright (>2000): {np.count_nonzero(img > 2000)}")
        
        if img.max() > 1000:
            max_loc = np.unravel_index(img.argmax(), img.shape)
            print(f"      Max intensity location: {max_loc} (row, col)")
            
            # Convert to stage coords
            max_Y, max_Z = self.camera.pixel_to_stage(max_loc[1], max_loc[0])
            print(f"      Max location (stage): ({max_Y/1000:.1f}, {max_Z/1000:.1f}) µm")
            print(f"      Distance from target: {np.sqrt((max_Y - tl_stage[0])**2 + (max_Z - tl_stage[1])**2)/1000:.1f} µm")
        
        # Calculate expected pixel position
        tl_pixel_expected = self.camera.stage_to_pixel(*tl_stage)
        print(f"\n   Expected pixel position: {tl_pixel_expected}")
        print(f"   Image center: ({self.camera.pixel_width//2}, {self.camera.pixel_height//2})")
        
        return img, tl_stage, tl_pixel_expected
    
    def test_cv_detection(self, img, tl_stage, tl_pixel_expected):
        """Test CV detection on rendered image."""
        print("\n" + "="*70)
        print("TEST: CV Detection")
        print("="*70)
        
        search_radius = 150
        print(f"\n🔍 Running fiducial detection:")
        print(f"   Expected pixel position: {tl_pixel_expected}")
        print(f"   Search radius: {search_radius} pixels")
        
        result = self.vt.find_fiducial_auto(
            img, 
            tl_pixel_expected, 
            search_radius=search_radius
        )
        
        if result:
            error_px = np.sqrt(
                (result['position'][0] - tl_pixel_expected[0])**2 +
                (result['position'][1] - tl_pixel_expected[1])**2
            )
            
            print(f"\n   ✅ FIDUCIAL FOUND!")
            print(f"      Method: {result['method']}")
            print(f"      Position: {result['position']} px")
            print(f"      Confidence: {result['confidence']:.3f}")
            print(f"      Pixel error: {error_px:.2f} pixels")
            
            # Convert back to stage coordinates
            found_Y, found_Z = self.camera.pixel_to_stage(*result['position'])
            print(f"\n      Stage coords: ({found_Y/1000:.3f}, {found_Z/1000:.3f}) µm")
            
            error_Y = abs(found_Y - tl_stage[0])
            error_Z = abs(found_Z - tl_stage[1])
            error_total = np.sqrt(error_Y**2 + error_Z**2)
            
            print(f"      Stage error: ({error_Y:.1f}, {error_Z:.1f}) nm")
            print(f"      Total error: {error_total:.1f} nm ({error_total/1000:.3f} µm)")
            
            if error_total < 5000:  # Less than 5 µm
                print(f"\n      ✅ Detection accuracy: EXCELLENT")
            elif error_total < 10000:  # Less than 10 µm
                print(f"\n      ⚠️  Detection accuracy: ACCEPTABLE")
            else:
                print(f"\n      ❌ Detection accuracy: POOR")
            
            return result
        else:
            print(f"\n   ❌ FIDUCIAL NOT FOUND")
            print(f"      Try adjusting search_radius or check image rendering")
            return None
    
    def visualize_results(self, img, tl_stage, tl_pixel_expected, detection_result):
        """Visualize camera image and detection results."""
        print("\n" + "="*70)
        print("VISUALIZATION")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Step 3 Debug - Camera Rendering & Detection', 
                     fontsize=16, fontweight='bold')
        
        # --- Plot 1: Full camera image ---
        ax1 = axes[0, 0]
        im1 = ax1.imshow(img, cmap='gray', vmin=0, vmax=3500, origin='lower')
        ax1.plot(tl_pixel_expected[0], tl_pixel_expected[1], 'r+', 
                markersize=30, markeredgewidth=3, label='Expected position')
        
        if detection_result:
            ax1.plot(detection_result['position'][0], detection_result['position'][1], 
                    'go', markersize=20, markerfacecolor='none', 
                    markeredgewidth=3, label='Detected position')
        
        ax1.set_title('Full Camera Image', fontweight='bold')
        ax1.set_xlabel('X pixels')
        ax1.set_ylabel('Y pixels')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(im1, ax=ax1, label='Intensity')
        
        # --- Plot 2: Zoomed region around expected position ---
        ax2 = axes[0, 1]
        zoom_size = 200
        x_center, y_center = tl_pixel_expected
        
        x_min = max(0, x_center - zoom_size)
        x_max = min(img.shape[1], x_center + zoom_size)
        y_min = max(0, y_center - zoom_size)
        y_max = min(img.shape[0], y_center + zoom_size)
        
        zoomed = img[y_min:y_max, x_min:x_max]
        im2 = ax2.imshow(zoomed, cmap='gray', vmin=0, vmax=3500, origin='lower',
                        extent=[x_min, x_max, y_min, y_max])
        
        ax2.plot(tl_pixel_expected[0], tl_pixel_expected[1], 'r+', 
                markersize=25, markeredgewidth=2)
        
        if detection_result:
            ax2.plot(detection_result['position'][0], detection_result['position'][1], 
                    'go', markersize=15, markerfacecolor='none', markeredgewidth=2)
        
        ax2.set_title(f'Zoomed Region ({zoom_size*2}x{zoom_size*2} px)', fontweight='bold')
        ax2.set_xlabel('X pixels')
        ax2.set_ylabel('Y pixels')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(im2, ax=ax2, label='Intensity')
        
        # --- Plot 3: Intensity histogram ---
        ax3 = axes[1, 0]
        hist, bins = np.histogram(img.flatten(), bins=100, range=(0, 4000))
        ax3.semilogy(bins[:-1], hist + 1, 'b-', linewidth=2)
        ax3.axvline(x=1000, color='r', linestyle='--', label='Threshold ~1000')
        ax3.axvline(x=2000, color='g', linestyle='--', label='Fiducials ~2000-3000')
        ax3.set_title('Intensity Histogram (log scale)', fontweight='bold')
        ax3.set_xlabel('Intensity')
        ax3.set_ylabel('Count (log)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # --- Plot 4: Text summary ---
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary = f"""
CAMERA CONFIGURATION
{'─'*50}
Resolution:     {self.camera.pixel_width} x {self.camera.pixel_height} pixels
Scale:          {self.camera.nm_per_pixel} nm/pixel
FOV:            {self.camera.fov_width_nm/1000:.1f} x {self.camera.fov_height_nm/1000:.1f} µm



IMAGE STATISTICS
{'─'*50}
Shape:          {img.shape}
Range:          [{img.min()}, {img.max()}]
Mean:           {img.mean():.1f}
Bright pixels:  {np.count_nonzero(img > 1000)} (>{1000})
Very bright:    {np.count_nonzero(img > 2000)} (>{2000})

DETECTION RESULTS
{'─'*50}
"""
        
        if detection_result:
            found_Y, found_Z = self.camera.pixel_to_stage(*detection_result['position'])
            error_total = np.sqrt((found_Y - tl_stage[0])**2 + (found_Z - tl_stage[1])**2)
            
            summary += f"""Status:         ✅ FOUND
Method:         {detection_result['method']}
Position:       {detection_result['position']}
Confidence:     {detection_result['confidence']:.4f}
Stage coords:   ({found_Y/1000:.3f}, {found_Z/1000:.3f}) µm
Total error:    {error_total:.1f} nm ({error_total/1000:.3f} µm)
"""
        else:
            summary += """Status:         ❌ NOT FOUND
Recommendation: Check if fiducials are visible in image
                Adjust search_radius or rotation angle
"""
        
        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Get block1 for summary
        block1 = self.layout_config['blocks'][1]
        
        plt.tight_layout()
        plt.savefig('step3_debug_results.png', dpi=150, bbox_inches='tight')
        print("\n   💾 Saved: step3_debug_results.png")
        plt.show()
    
    def run_full_debug(self):
        """Run complete debug sequence."""
        print("\n" + "="*70)
        print("STEP 3 FULL DEBUG SEQUENCE")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  ASCII file: {self.ascii_file}")
        print(f"  Rotation: {self.rotation_angle}°")
        print(f"  Translation: {self.translation} nm")
        
        # 1. Setup
        if not self.setup():
            return False
        
        # 2. Verify coordinates
        if not self.verify_coordinate_system():
            print("\n❌ Coordinate system verification failed!")
            return False
        
        # 3. Test camera rendering
        img, tl_stage, tl_pixel_expected = self.test_camera_rendering()
        
        if img.max() < 1000:
            print("\n⚠️  WARNING: Image appears to have no bright features!")
            print("   This suggests fiducials are not being rendered.")
        
        # 4. Test CV detection
        detection_result = self.test_cv_detection(img, tl_stage, tl_pixel_expected)
        
        # 5. Visualize
        self.visualize_results(img, tl_stage, tl_pixel_expected, detection_result)
        
        # Final status
        print("\n" + "="*70)
        if detection_result:
            print("✅ STEP 3 DEBUG COMPLETE - Detection successful!")
        else:
            print("❌ STEP 3 DEBUG COMPLETE - Detection failed, see diagnostics above")
        print("="*70)
        
        return detection_result is not None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug Step 3 - Camera and CV detection")
    parser.add_argument('--ascii', type=str, default="./AlignmentSystem/ascii_sample.ASC",
                       help='Path to ASCII file')
    parser.add_argument('--rotation', type=float, default=3.0,
                       help='Simulated rotation angle in degrees (default: 3.0)')
    parser.add_argument('--no-rotation', action='store_true',
                       help='Set rotation to 0 for simpler debugging')
    
    args = parser.parse_args()
    
    # Create debugger
    debugger = Step3Debugger(ascii_file=args.ascii)
    
    if args.no_rotation:
        debugger.rotation_angle = 0.0
        print("⚠️  Rotation set to 0° for debugging")
    else:
        debugger.rotation_angle = args.rotation
    
    # Run full debug
    success = debugger.run_full_debug()
    
    if success:
        print("\n🎉 Debug completed successfully!")
        print("   Review 'step3_debug_results.png' for detailed visualization")
    else:
        print("\n💡 Troubleshooting tips:")
        print("   1. Try running with --no-rotation to test without rotation")
        print("   2. Check that ASCII file contains marker definitions")
        print("   3. Verify camera FOV is large enough to see fiducials")
        print("   4. Check console output for coordinate mismatches")


if __name__ == "__main__":
    main()