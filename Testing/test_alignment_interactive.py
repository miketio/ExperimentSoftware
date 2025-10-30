#!/usr/bin/env python3
# test_alignment_interactive.py
"""
Interactive testing script for alignment system with visualization.
Tests each component step-by-step with plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
from AlignmentSystem.cv_tools import VisionTools
from AlignmentSystem.camera_simulation import Camera

class AlignmentSystemTester:
    """Interactive tester for alignment system."""
    
    def __init__(self, ascii_file: str = "./AlignmentSystem/ascii_sample.ASC"):
        self.ascii_file = ascii_file
        self.parsed_data = None
        self.layout_config = None
        self.measured_corners = {}  # Store measured corner positions (Y, Z) in nm
        self.transform = None
        self.rotation_angle = 3.0  # Simulated rotation in degrees
        self.translation = (10000, 0)  # Simulated translation (Y, Z) in nm
        self.camera = Camera(pixel_width=2048, pixel_height=2048, nm_per_pixel=1000)
        
    def design_to_stage(self, u_um, v_um):
        """
        Convert design coordinates (u, v) in µm to stage coordinates (Y, Z) in nm.
        Applies rotation and translation.
        """
        # Convert to nm
        u_nm = u_um * 1000
        v_nm = v_um * 1000
        
        # Apply rotation
        angle_rad = np.radians(self.rotation_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        Y = cos_a * u_nm - sin_a * v_nm + self.translation[0]
        Z = sin_a * u_nm + cos_a * v_nm + self.translation[1]
        
        return (Y, Z)
    
    def step1_test_ascii_parser(self):
        """Step 1: Parse ASCII and plot the block."""
        print("\n" + "="*70)
        print("STEP 1: ASCII PARSER - Plot Single Block")
        print("="*70)
        
        from AlignmentSystem.ascii_parser import ASCIIParser
        
        if not Path(self.ascii_file).exists():
            print(f"❌ ASCII file not found: {self.ascii_file}")
            return False
        
        try:
            parser = ASCIIParser(self.ascii_file)
            self.parsed_data = parser.parse()
            
            print(f"✅ Parsed: {self.ascii_file}")
            print(f"   Markers: {len(self.parsed_data['markers'])}")
            print(f"   Waveguides: {len(self.parsed_data['waveguides'])}")
            print(f"   Gratings: {len(self.parsed_data['gratings'])}")
            
            # Plot single block
            self._plot_single_block()
            
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _plot_single_block(self):
        """Plot a single block from parsed data."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot markers (corners)
        for marker in self.parsed_data['markers']:
            pos = marker['position']
            corner = marker['corner']
            coords = marker['coords']
            
            # Plot marker polygon
            poly = patches.Polygon(coords, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(poly)
            
            # Label corner
            ax.plot(pos[0], pos[1], 'ro', markersize=10)
            ax.text(pos[0], pos[1] + 5, corner, ha='center', fontsize=8, color='red')
        
        # Plot waveguides
        for wg in self.parsed_data['waveguides']:
            u_start = wg['u_start']
            u_end = wg['u_end']
            v_center = wg['v_center']
            width = wg['width']
            
            # Draw waveguide as rectangle
            rect = patches.Rectangle(
                (u_start, v_center - width/2),
                u_end - u_start,
                width,
                fill=True,
                facecolor='lightblue',
                edgecolor='blue',
                alpha=0.5
            )
            ax.add_patch(rect)
            
            # Label waveguide number
            if wg['number'] % 5 == 0:  # Label every 5th
                ax.text(u_start - 5, v_center, f"WG{wg['number']}", 
                       ha='right', va='center', fontsize=6)
        
        # Plot gratings
        for grating in self.parsed_data['gratings']:
            pos = grating['position']
            side = grating['side']
            
            color = 'green' if side == 'left' else 'orange'
            ax.plot(pos[0], pos[1], 'o', color=color, markersize=8, alpha=0.7)
        
        # Find and highlight WG25
        wg25 = None
        for wg in self.parsed_data['waveguides']:
            if wg['number'] == 25:
                wg25 = wg
                break
        
        if wg25:
            # Highlight WG25
            rect = patches.Rectangle(
                (wg25['u_start'], wg25['v_center'] - wg25['width']/2),
                wg25['u_end'] - wg25['u_start'],
                wg25['width'],
                fill=False,
                edgecolor='red',
                linewidth=3
            )
            ax.add_patch(rect)
            ax.text(wg25['u_start'] - 10, wg25['v_center'], 
                   "WG25", ha='right', va='center', fontsize=10, 
                   color='red', weight='bold')
        
        ax.set_xlabel('u (µm)', fontsize=12)
        ax.set_ylabel('v (µm)', fontsize=12)
        ax.set_title('Single Block Layout (Design Coordinates)', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('test_1_single_block.png', dpi=150)
        print("   📊 Saved plot: test_1_single_block.png")
        plt.show()
    
    def step2_test_layout_generator(self):
        """Step 2: Generate layout with rotation and translation, then plot."""
        print("\n" + "="*70)
        print("STEP 2: LAYOUT GENERATOR - Apply Rotation & Translation")
        print("="*70)
        
        from AlignmentSystem.layout_config_generator import generate_layout_config
        
        try:
            # Generate layout config
            config_file = "config/test_layout.json"
            Path("config").mkdir(exist_ok=True)
            
            self.layout_config = generate_layout_config(self.ascii_file, config_file)
            
            print(f"✅ Generated layout config")
            print(f"   Total blocks: {self.layout_config['total_blocks']}")
            print(f"   Block spacing: {self.layout_config['block_spacing']} µm")
            
            # Simulate rotation and translation
            print(f"\n🔄 Simulating sample rotation: {self.rotation_angle}°")
            print(f"🔄 Simulating translation: {self.translation} nm")
            
            # Plot rotated and translated layout
            self._plot_rotated_layout()
            
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _plot_rotated_layout(self):
        """Plot the full layout with rotation and translation applied."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Plot all blocks
        for block_id, block in self.layout_config['blocks'].items():
            center = block['center']
            
            # Get block corners in design coordinates
            corners_design = [
                block['fiducials']['top_left'],
                block['fiducials']['top_right'],
                block['fiducials']['bottom_right'],
                block['fiducials']['bottom_left'],
                block['fiducials']['top_left']  # Close polygon
            ]
            
            # Apply rotation and translation
            corners_stage = [self.design_to_stage(u, v) for u, v in corners_design]
            
            # Convert to arrays for plotting
            Y_coords = [c[0]/1000 for c in corners_stage]  # Convert to µm for plotting
            Z_coords = [c[1]/1000 for c in corners_stage]
            
            # Plot block outline
            ax.plot(Y_coords, Z_coords, 'b-', linewidth=1, alpha=0.5)
            
            # Plot block center
            center_stage = self.design_to_stage(*center)
            ax.plot(center_stage[0]/1000, center_stage[1]/1000, 'ko', markersize=3)
            
            # Label every 5th block
            if block_id % 5 == 0:
                ax.text(center_stage[0]/1000, center_stage[1]/1000, 
                       f"B{block_id}", ha='center', va='center', 
                       fontsize=6, color='blue')
        
        # Highlight block 10
        block10 = self.layout_config['blocks'][10]
        corners_design = [
            block10['fiducials']['top_left'],
            block10['fiducials']['top_right'],
            block10['fiducials']['bottom_right'],
            block10['fiducials']['bottom_left'],
            block10['fiducials']['top_left']
        ]
        corners_stage = [self.design_to_stage(u, v) for u, v in corners_design]
        Y_coords = [c[0]/1000 for c in corners_stage]
        Z_coords = [c[1]/1000 for c in corners_stage]
        
        ax.plot(Y_coords, Z_coords, 'r-', linewidth=3, label='Block 10')
        
        # Mark block 10 center
        center_stage = self.design_to_stage(*block10['center'])
        ax.plot(center_stage[0]/1000, center_stage[1]/1000, 'r*', markersize=15)
        ax.text(center_stage[0]/1000, center_stage[1]/1000 + 20, 
               'Block 10', ha='center', fontsize=12, color='red', weight='bold')
        
        ax.set_xlabel('Y (µm) - Stage Coordinates', fontsize=12)
        ax.set_ylabel('Z (µm) - Stage Coordinates', fontsize=12)
        ax.set_title(f'Full Layout with Rotation ({self.rotation_angle}°) and Translation', 
                    fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('test_2_rotated_layout.png', dpi=150)
        print("   📊 Saved plot: test_2_rotated_layout.png")
        plt.show()


    def step3_test_cv_tools(self):
        """Step 3: Test CV tools - find first and last corners with camera simulation.
        FIXED VERSION with detailed debugging and error handling.
        """
        print("\n" + "="*70)
        print("STEP 3: COMPUTER VISION - Find Corner Fiducials with Camera")
        print("="*70)
        
        try:
            # Import VisionTools
            from AlignmentSystem.cv_tools import VisionTools
            vt = VisionTools()
            
            print(f"\n📷 Camera configuration:")
            print(f"   Resolution: {self.camera.pixel_width}x{self.camera.pixel_height} pixels")
            print(f"   Scale: {self.camera.nm_per_pixel} nm/pixel")
            print(f"   FOV: {self.camera.fov_width_nm/1000:.1f} x {self.camera.fov_height_nm/1000:.1f} µm")
            
            # Get target corners
            print(f"\n🎯 Getting target fiducial positions...")
            block1 = self.layout_config['blocks'][1]
            block20 = self.layout_config['blocks'][20]
            
            tl_design = block1['fiducials']['top_left']
            br_design = block20['fiducials']['bottom_right']
            
            print(f"   Block 1 TL (design): ({tl_design[0]:.1f}, {tl_design[1]:.1f}) µm")
            print(f"   Block 20 BR (design): ({br_design[0]:.1f}, {br_design[1]:.1f}) µm")
            
            # Convert to stage coordinates
            tl_stage = self.design_to_stage(*tl_design)
            br_stage = self.design_to_stage(*br_design)
            
            print(f"\n📍 Target positions (stage coords after rotation):")
            print(f"   Top-left: Y={tl_stage[0]:.0f} nm ({tl_stage[0]/1000:.1f} µm)")
            print(f"              Z={tl_stage[1]:.0f} nm ({tl_stage[1]/1000:.1f} µm)")
            print(f"   Bottom-right: Y={br_stage[0]:.0f} nm ({br_stage[0]/1000:.1f} µm)")
            print(f"                  Z={br_stage[1]:.0f} nm ({br_stage[1]/1000:.1f} µm)")
            
            # Build layout elements
            print(f"\n🎨 Building scene elements...")
            layout_elements = self._build_layout_elements_for_camera()
            
            print(f"   Total elements: {len(layout_elements)}")
            
            # Count by type
            type_counts = {}
            for elem in layout_elements:
                t = elem['type']
                type_counts[t] = type_counts.get(t, 0) + 1
            
            for elem_type, count in type_counts.items():
                print(f"      {elem_type}: {count}")
            
            # CAPTURE TOP-LEFT FIDUCIAL
            print(f"\n" + "-"*70)
            print("🔍 CAPTURING TOP-LEFT FIDUCIAL")
            print("-"*70)
            
            print(f"   Moving camera to Y={tl_stage[0]:.0f} nm, Z={tl_stage[1]:.0f} nm")
            self.camera.move_to(*tl_stage)
            
            print(f"   Taking image...")
            img_tl = self.camera.take_image(layout_elements, debug=True)
            
            print(f"   Image captured: {img_tl.shape}, dtype={img_tl.dtype}")
            print(f"   Image range: [{img_tl.min()}, {img_tl.max()}]")
            
            # Calculate expected pixel position (should be near center)
            tl_pixel_expected = self.camera.stage_to_pixel(*tl_stage)
            print(f"\n   Expected pixel position: ({tl_pixel_expected[0]}, {tl_pixel_expected[1]})")
            print(f"   Image center: ({self.camera.pixel_width//2}, {self.camera.pixel_height//2})")
            
            # Run detection
            search_radius = 150
            print(f"   Running detection with search radius {search_radius}px...")
            result_tl = vt.find_fiducial_auto(img_tl, tl_pixel_expected, search_radius=search_radius)
            
            if result_tl:
                error_px = np.sqrt((result_tl['position'][0] - tl_pixel_expected[0])**2 +
                                (result_tl['position'][1] - tl_pixel_expected[1])**2)
                
                print(f"\n   ✅ TOP-LEFT FIDUCIAL FOUND!")
                print(f"      Method: {result_tl['method']}")
                print(f"      Position: ({result_tl['position'][0]}, {result_tl['position'][1]}) px")
                print(f"      Confidence: {result_tl['confidence']:.3f}")
                print(f"      Error: {error_px:.2f} pixels")
                
                # Convert back to stage coordinates
                found_Y, found_Z = self.camera.pixel_to_stage(*result_tl['position'])
                self.measured_corners['top_left'] = (int(found_Y), int(found_Z))
                
                print(f"      Stage coords: Y={found_Y:.0f} nm, Z={found_Z:.0f} nm")
                print(f"      Error: Y={abs(found_Y - tl_stage[0]):.0f} nm, Z={abs(found_Z - tl_stage[1]):.0f} nm")
            else:
                print(f"\n   ❌ TOP-LEFT FIDUCIAL NOT FOUND")
                print(f"      Check if fiducial is visible in camera FOV")
                return False
            
            # CAPTURE BOTTOM-RIGHT FIDUCIAL
            print(f"\n" + "-"*70)
            print("🔍 CAPTURING BOTTOM-RIGHT FIDUCIAL")
            print("-"*70)
            
            print(f"   Moving camera to Y={br_stage[0]:.0f} nm, Z={br_stage[1]:.0f} nm")
            self.camera.move_to(*br_stage)
            
            print(f"   Taking image...")
            img_br = self.camera.take_image(layout_elements, debug=True)
            
            print(f"   Image captured: {img_br.shape}, dtype={img_br.dtype}")
            print(f"   Image range: [{img_br.min()}, {img_br.max()}]")
            
            # Calculate expected pixel position
            br_pixel_expected = self.camera.stage_to_pixel(*br_stage)
            print(f"\n   Expected pixel position: ({br_pixel_expected[0]}, {br_pixel_expected[1]})")
            
            # Run detection
            print(f"   Running detection with search radius {search_radius}px...")
            result_br = vt.find_fiducial_auto(img_br, br_pixel_expected, search_radius=search_radius)
            
            if result_br:
                error_px = np.sqrt((result_br['position'][0] - br_pixel_expected[0])**2 +
                                (result_br['position'][1] - br_pixel_expected[1])**2)
                
                print(f"\n   ✅ BOTTOM-RIGHT FIDUCIAL FOUND!")
                print(f"      Method: {result_br['method']}")
                print(f"      Position: ({result_br['position'][0]}, {result_br['position'][1]}) px")
                print(f"      Confidence: {result_br['confidence']:.3f}")
                print(f"      Error: {error_px:.2f} pixels")
                
                # Convert back to stage coordinates
                found_Y, found_Z = self.camera.pixel_to_stage(*result_br['position'])
                self.measured_corners['bottom_right'] = (int(found_Y), int(found_Z))
                
                print(f"      Stage coords: Y={found_Y:.0f} nm, Z={found_Z:.0f} nm")
                print(f"      Error: Y={abs(found_Y - br_stage[0]):.0f} nm, Z={abs(found_Z - br_stage[1]):.0f} nm")
            else:
                print(f"\n   ❌ BOTTOM-RIGHT FIDUCIAL NOT FOUND")
                print(f"      Check if fiducial is visible in camera FOV")
                return False
            
            # PLOT RESULTS
            print(f"\n📊 Creating visualization...")
            self._plot_detection_results_simple(
                img_tl, img_br,
                tl_stage, br_stage,
                tl_pixel_expected, br_pixel_expected,
                result_tl, result_br
            )
            
            print(f"\n" + "="*70)
            print("✅ STEP 3 COMPLETE - Both fiducials detected!")
            print("="*70)
            
            return True
            
        except Exception as e:
            print(f"\n❌ STEP 3 FAILED")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            return False


    def _build_layout_elements_for_camera(self):
        """
        Builds layout_elements list from self.layout_config and self.parsed_data.
        FIXED VERSION with proper coordinate handling.
        
        Returns:
            list of element dicts compatible with Camera.take_image
        """
        import numpy as np
        
        layout_elements = []
        
        print(f"\n   Building elements for {len(self.layout_config['blocks'])} blocks...")
        
        for block_id, block in self.layout_config['blocks'].items():
            # Get block center in design coordinates
            block_center_design = block['center']
            
            # Add fiducials (corner markers)
            for corner_name, corner_design in block['fiducials'].items():
                # corner_design is (u, v) in µm relative to origin
                # Convert to stage coordinates (returns nm)
                corner_stage_nm = self.design_to_stage(*corner_design)
                
                # Store as µm for camera (camera expects µm in coords_stage)
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
            
            # Convert each corner to stage coordinates
            corners_stage_um = []
            for u, v in corners_design:
                corner_stage_nm = self.design_to_stage(u, v)
                corners_stage_um.append((corner_stage_nm[0] / 1000.0, corner_stage_nm[1] / 1000.0))
            
            layout_elements.append({
                'type': 'block_outline',
                'coords_stage': corners_stage_um,  # List of (Y, Z) in µm
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
                # print(f"      Grating '{name}' in Block {block_id}: Design coords (u={u_global:.1f}, v={v_global:.1f}) µm")
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
        
        print(f"   ✅ Built {len(layout_elements)} elements")
        
        return layout_elements


    def _plot_detection_results_simple(self, img_tl, img_br, 
                                    tl_stage, br_stage,
                                    tl_pixel_expected, br_pixel_expected,
                                    result_tl, result_br):
        """Simplified plotting for faster debugging."""
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Fiducial Detection Results', fontsize=16, fontweight='bold')
        
        # Top-left image
        ax1.imshow(img_tl, cmap='gray', vmin=0, vmax=3500, origin='lower')
        ax1.plot(tl_pixel_expected[0], tl_pixel_expected[1], 'r+', markersize=20, 
                markeredgewidth=3, label='Expected')
        if result_tl:
            ax1.plot(result_tl['position'][0], result_tl['position'][1], 'go', 
                    markersize=15, markerfacecolor='none', markeredgewidth=2, label='Found')
        ax1.set_title('Top-Left Fiducial', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom-right image
        ax2.imshow(img_br, cmap='gray', vmin=0, vmax=3500, origin='lower')
        ax2.plot(br_pixel_expected[0], br_pixel_expected[1], 'b+', markersize=20,
                markeredgewidth=3, label='Expected')
        if result_br:
            ax2.plot(result_br['position'][0], result_br['position'][1], 'mo',
                    markersize=15, markerfacecolor='none', markeredgewidth=2, label='Found')
        ax2.set_title('Bottom-Right Fiducial', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Overview map
        ax3.set_title('Stage Overview', fontweight='bold')
        for block_id, block in self.layout_config['blocks'].items():
            corners = [
                block['fiducials']['top_left'],
                block['fiducials']['top_right'],
                block['fiducials']['bottom_right'],
                block['fiducials']['bottom_left'],
                block['fiducials']['top_left']
            ]
            corners_stage = [self.design_to_stage(u, v) for u, v in corners]
            Y_coords = [c[0]/1000 for c in corners_stage]
            Z_coords = [c[1]/1000 for c in corners_stage]
            ax3.plot(Y_coords, Z_coords, 'b-', linewidth=1, alpha=0.3)
        
        ax3.plot(tl_stage[0]/1000, tl_stage[1]/1000, 'r*', markersize=15, label='TL Target')
        ax3.plot(br_stage[0]/1000, br_stage[1]/1000, 'b*', markersize=15, label='BR Target')
        
        if result_tl:
            tl_found = self.measured_corners['top_left']
            ax3.plot(tl_found[0]/1000, tl_found[1]/1000, 'go', markersize=12,
                    markerfacecolor='none', markeredgewidth=2, label='Found')
        if result_br:
            br_found = self.measured_corners['bottom_right']
            ax3.plot(br_found[0]/1000, br_found[1]/1000, 'mo', markersize=12,
                    markerfacecolor='none', markeredgewidth=2)
        
        ax3.set_xlabel('Y (µm)')
        ax3.set_ylabel('Z (µm)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        # Results summary
        ax4.axis('off')
        summary_text = f"""
        DETECTION SUMMARY

        Top-Left Fiducial:
        Method: {result_tl['method'] if result_tl else 'N/A'}
        Confidence: {f"{result_tl['confidence']:.3f}" if result_tl else 'N/A'}
        Position: {result_tl['position'] if result_tl else 'NOT FOUND'}

        Bottom-Right Fiducial:
        Method: {result_br['method'] if result_br else 'N/A'}
        Confidence: {f"{result_br['confidence']:.3f}" if result_br else 'N/A'}
        Position: {result_br['position'] if result_br else 'NOT FOUND'}

        Status: {'✅ BOTH FOUND' if result_tl and result_br else '❌ DETECTION FAILED'}
        """

        
        ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('test_3_camera_detection.png', dpi=150, bbox_inches='tight')
        print(f"   💾 Saved: test_3_camera_detection.png")
        plt.show()

    def step4_coordinate_transform(self):
        """Step 4: Calibrate coordinate transformation using measured corners."""
        print("\n" + "="*70)
        print("STEP 4: COORDINATE TRANSFORMATION - Calibrate")
        print("="*70)
        
        from AlignmentSystem.coordinate_transform import CoordinateTransform
        
        try:
            if not self.measured_corners:
                print("❌ No measured corners available. Run step 3 first.")
                return False
            
            self.transform = CoordinateTransform()
            
            # Get design coordinates
            block1 = self.layout_config['blocks'][1]
            block20 = self.layout_config['blocks'][20]
            
            design_tl = block1['fiducials']['top_left']
            design_br = block20['fiducials']['bottom_right']
            
            # Get measured coordinates
            measured_tl = self.measured_corners['top_left']
            measured_br = self.measured_corners['bottom_right']
            
            print(f"\n📍 Calibration points:")
            print(f"   Point 1 (Top-Left):")
            print(f"      Design: ({design_tl[0]:.1f}, {design_tl[1]:.1f}) µm")
            print(f"      Measured: ({measured_tl[0]}, {measured_tl[1]}) nm")
            print(f"   Point 2 (Bottom-Right):")
            print(f"      Design: ({design_br[0]:.1f}, {design_br[1]:.1f}) µm")
            print(f"      Measured: ({measured_br[0]}, {measured_br[1]}) nm")
            
            # Calibrate
            result = self.transform.calibrate(
                measured_points=[measured_tl, measured_br],
                design_points=[design_tl, design_br]
            )
            
            print(f"\n✅ Calibration successful:")
            print(f"   Method: {result['method']}")
            print(f"   Detected angle: {result['angle_deg']:.3f}°")
            print(f"   Expected angle: {self.rotation_angle}°")
            print(f"   Angle error: {abs(result['angle_deg'] - self.rotation_angle):.3f}°")
            print(f"   Translation: ({result['translation_nm'][0]:.0f}, {result['translation_nm'][1]:.0f}) nm")
            print(f"   Mean positioning error: {result['mean_error_nm']:.3f} nm")
            
            # Verify transformation
            self._verify_transformation()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _verify_transformation(self):
        """Verify transformation with test points."""
        print("\n🔬 Verifying transformation with test points...")
        
        # Test several points across the layout
        test_points = [
            (100, 100, "Corner"),
            (700, 300, "Middle"),
            (1300, 500, "Far corner")
        ]
        
        for u, v, label in test_points:
            stage_coords = self.transform.design_to_stage(u, v)
            back_coords = self.transform.stage_to_design(*stage_coords)
            
            error = np.sqrt((back_coords[0] - u)**2 + (back_coords[1] - v)**2)
            
            print(f"   {label}: ({u}, {v}) µm")
            print(f"      → Stage: ({stage_coords[0]}, {stage_coords[1]}) nm")
            print(f"      → Back: ({back_coords[0]:.3f}, {back_coords[1]:.3f}) µm")
            print(f"      → Round-trip error: {error:.6f} µm")
    
    def step5_find_wg25_and_plot(self):
        """Step 5: Find WG25 from block 10 and plot final result."""
        print("\n" + "="*70)
        print("STEP 5: LOCATE WG25 - Block 10")
        print("="*70)
        
        from AlignmentSystem.ascii_parser import find_waveguide_grating
        
        try:
            if not self.transform:
                print("❌ Coordinate transform not calibrated. Run step 4 first.")
                return False
            
            # Get Block 10 info
            block10 = self.layout_config['blocks'][10]
            ref_coord= block10['fiducials']['bottom_left']
            # Get WG25 grating position (left side) in design coordinates
            wg25_left_design = block10['gratings']['wg25_left']
            wg25_left_design = (ref_coord[0] + wg25_left_design[0],
                                 ref_coord[1] + wg25_left_design[1])
            print(f"\n📍 Target: Block 10, Waveguide 25, LEFT grating")
            print(f"   Design coordinates: ({wg25_left_design[0]:.3f}, {wg25_left_design[1]:.3f}) µm")
            
            # Transform to stage coordinates
            wg25_stage = self.transform.design_to_stage(*wg25_left_design)
            
            print(f"   Stage coordinates: ({wg25_stage[0]}, {wg25_stage[1]}) nm")
            print(f"   Stage coordinates: ({wg25_stage[0]/1000:.1f}, {wg25_stage[1]/1000:.1f}) µm")
            
            # Plot everything
            self._plot_final_result(wg25_left_design, wg25_stage)
            
            print(f"\n✅ All tests completed successfully!")
            print(f"\n📋 Summary:")
            print(f"   1. ✅ ASCII parsed - {len(self.parsed_data['waveguides'])} waveguides found")
            print(f"   2. ✅ Layout generated - {self.layout_config['total_blocks']} blocks")
            print(f"   3. ✅ Fiducials detected - 2/2 found")
            print(f"   4. ✅ Transform calibrated - {self.transform.angle_deg:.2f}° rotation")
            print(f"   5. ✅ WG25 located - Ready for alignment")
            
            self.step6_verify_wg25_with_camera(wg25_left_design, wg25_stage)
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _plot_final_result(self, wg25_design, wg25_stage):
        """Plot final result showing rotated layout with WG25 target."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # === LEFT PLOT: Design coordinates with Block 10 ===
        ax1.set_title('Design Coordinates (Original)', fontsize=14, weight='bold')
        
        # Plot all blocks
        for block_id, block in self.layout_config['blocks'].items():
            corners = [
                block['fiducials']['top_left'],
                block['fiducials']['top_right'],
                block['fiducials']['bottom_right'],
                block['fiducials']['bottom_left'],
                block['fiducials']['top_left']
            ]
            u_coords = [c[0] for c in corners]
            v_coords = [c[1] for c in corners]
            
            if block_id == 10:
                ax1.plot(u_coords, v_coords, 'r-', linewidth=3, label='Block 10')
            else:
                ax1.plot(u_coords, v_coords, 'b-', linewidth=1, alpha=0.3)
        
        # Highlight Block 10 center
        block10 = self.layout_config['blocks'][10]
        ax1.plot(block10['center'][0], block10['center'][1], 'r*', markersize=15)
        
        # Mark WG25 grating
        ax1.plot(wg25_design[0], wg25_design[1], 'gX', markersize=20, markeredgewidth=3,
                label='WG25 Left Grating')
        ax1.text(wg25_design[0], wg25_design[1] + 10, 'WG25\nTarget', 
                ha='center', fontsize=12, color='green', weight='bold')
        
        ax1.set_xlabel('u (µm)', fontsize=12)
        ax1.set_ylabel('v (µm)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_aspect('equal')
        
        # === RIGHT PLOT: Rotated stage coordinates ===
        ax2.set_title(f'Stage Coordinates (Rotated {self.rotation_angle}°)', 
                     fontsize=14, weight='bold')
        
        # Plot all blocks in stage coordinates
        for block_id, block in self.layout_config['blocks'].items():
            corners_design = [
                block['fiducials']['top_left'],
                block['fiducials']['top_right'],
                block['fiducials']['bottom_right'],
                block['fiducials']['bottom_left'],
                block['fiducials']['top_left']
            ]
            corners_stage = [self.transform.design_to_stage(u, v) for u, v in corners_design]
            
            Y_coords = [c[0]/1000 for c in corners_stage]  # Convert to µm
            Z_coords = [c[1]/1000 for c in corners_stage]
            
            if block_id == 10:
                ax2.plot(Y_coords, Z_coords, 'r-', linewidth=3, label='Block 10')
            else:
                ax2.plot(Y_coords, Z_coords, 'b-', linewidth=1, alpha=0.3)
        
        # Highlight Block 10 center
        center_stage = self.transform.design_to_stage(*block10['center'])
        ax2.plot(center_stage[0]/1000, center_stage[1]/1000, 'r*', markersize=15)
        
        # Mark WG25 grating in stage coordinates
        ax2.plot(wg25_stage[0]/1000, wg25_stage[1]/1000, 'gX', 
                markersize=20, markeredgewidth=3, label='WG25 Left Grating')
        ax2.text(wg25_stage[0]/1000, wg25_stage[1]/1000 + 10, 'WG25\nTarget', 
                ha='center', fontsize=12, color='green', weight='bold')
        
        # Mark measured fiducials
        if self.measured_corners:
            tl = self.measured_corners.get('top_left')
            br = self.measured_corners.get('bottom_right')
            if tl:
                ax2.plot(tl[0]/1000, tl[1]/1000, 'mo', markersize=10, 
                        label='Measured Fiducials')
            if br:
                ax2.plot(br[0]/1000, br[1]/1000, 'mo', markersize=10)
        
        ax2.set_xlabel('Y (µm) - Stage Coordinates', fontsize=12)
        ax2.set_ylabel('Z (µm) - Stage Coordinates', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('test_5_final_wg25_location.png', dpi=150)
        print("   📊 Saved plot: test_5_final_wg25_location.png")
        plt.show()

    def step6_verify_wg25_with_camera(self, wg25_design, wg25_stage):
        """Step 6: Move camera to Block 10 and verify WG25 grating visibility."""
        print("\n" + "="*70)
        print("STEP 6: CAMERA VERIFICATION - Block 10 WG25")
        print("="*70)
        
        from AlignmentSystem.cv_tools import VisionTools
        vt = VisionTools()
        
        print(f"\n📷 Moving camera to WG25 left grating...")
        print(f"   Target stage coords: ({wg25_stage[0]}, {wg25_stage[1]}) nm")
        print(f"   Target stage coords: ({wg25_stage[0]/1000:.1f}, {wg25_stage[1]/1000:.1f}) µm")
        
        # Move camera to WG25 position
        self.camera.move_to(*wg25_stage)
        
        # Build layout elements
        print(f"\n🎨 Rendering scene...")
        layout_elements = self._build_layout_elements_for_camera()
        
        # Take image
        print(f"   Taking image...")
        img = self.camera.take_image(layout_elements, debug=True)
        
        print(f"\n   Image captured: {img.shape}, dtype={img.dtype}")
        print(f"   Image range: [{img.min()}, {img.max()}]")
        print(f"   Bright pixels (>1000): {np.count_nonzero(img > 1000)}")
        
        # Calculate expected pixel position (should be near center)
        wg25_pixel = self.camera.stage_to_pixel(*wg25_stage)
        print(f"\n   Expected WG25 pixel position: ({wg25_pixel[0]}, {wg25_pixel[1]})")
        print(f"   Image center: ({self.camera.pixel_width//2}, {self.camera.pixel_height//2})")
        
        # Visualize
        self._plot_wg25_camera_view(img, wg25_pixel, wg25_stage)
        
        print(f"\n✅ STEP 6 COMPLETE - Camera positioned at WG25")
        return True

    def _plot_wg25_camera_view(self, img, wg25_pixel, wg25_stage):
        """Plot camera view centered on WG25."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle('Block 10 - WG25 Left Grating Camera View', fontsize=16, fontweight='bold')
        
        # Full camera image
        ax1 = axes[0]
        im1 = ax1.imshow(img, cmap='gray', vmin=0, vmax=3500, origin='lower')
        ax1.plot(wg25_pixel[0], wg25_pixel[1], 'gX', markersize=25, markeredgewidth=3, 
                label='WG25 Target')
        ax1.plot(self.camera.pixel_width//4, self.camera.pixel_height//4, 'r+', 
                markersize=20, markeredgewidth=2, label='Camera Center')
        ax1.set_title('Full Camera View', fontweight='bold')
        ax1.set_xlabel('X pixels')
        ax1.set_ylabel('Y pixels')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(im1, ax=ax1, label='Intensity')
        
        # Zoomed view around WG25
        ax2 = axes[1]
        zoom_size = 300
        x_center, y_center = wg25_pixel
        
        x_min = max(0, x_center - zoom_size)
        x_max = min(img.shape[1], x_center + zoom_size)
        y_min = max(0, y_center - zoom_size)
        y_max = min(img.shape[0], y_center + zoom_size)
        
        zoomed = img[y_min:y_max, x_min:x_max]
        im2 = ax2.imshow(zoomed, cmap='gray', vmin=0, vmax=3500, origin='lower',
                        extent=[x_min, x_max, y_min, y_max])
        
        ax2.plot(wg25_pixel[0], wg25_pixel[1], 'gX', markersize=20, markeredgewidth=3)
        ax2.set_title(f'Zoomed View ({zoom_size*2}x{zoom_size*2} px)', fontweight='bold')
        ax2.set_xlabel('X pixels')
        ax2.set_ylabel('Y pixels')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(im2, ax=ax2, label='Intensity')
        
        # Summary text
        ax3 = axes[2]
        ax3.axis('off')
        
        summary = f"""
    CAMERA VIEW SUMMARY
    {'─'*50}

    Target: Block 10, Waveguide 25, LEFT grating

    Stage Position:
    Y = {wg25_stage[0]:>10.0f} nm ({wg25_stage[0]/1000:>7.1f} µm)
    Z = {wg25_stage[1]:>10.0f} nm ({wg25_stage[1]/1000:>7.1f} µm)

    Pixel Position:
    X = {wg25_pixel[0]:>4d} px
    Y = {wg25_pixel[1]:>4d} px

    Camera FOV:
    {self.camera.fov_width_nm/1000:.1f} × {self.camera.fov_height_nm/1000:.1f} µm
    {self.camera.pixel_width} × {self.camera.pixel_height} pixels
    Resolution: {self.camera.nm_per_pixel} nm/pixel

    Image Statistics:
    Range: [{img.min()}, {img.max()}]
    Mean: {img.mean():.1f}
    Bright pixels: {np.count_nonzero(img > 1000)}

    Status: ✅ Camera positioned at target
        """
        
        ax3.text(0.05, 0.95, summary, transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('test_6_wg25_camera_view.png', dpi=150, bbox_inches='tight')
        print(f"   💾 Saved: test_6_wg25_camera_view.png")
        plt.show()

    def run_all_tests(self):
        """Run all tests sequentially."""
        print("\n" + "="*70)
        print("ALIGNMENT SYSTEM INTERACTIVE TEST SUITE")
        print("="*70)
        print("\nThis will test all components with visualization:")
        print("  1. Parse ASCII and plot single block")
        print("  2. Generate layout with rotation/translation")
        print("  3. Find corner fiducials with CV and Camera")
        print("  4. Calibrate coordinate transformation")
        print("  5. Locate WG25 target in Block 10")
        print("="*70)
        
        input("\nPress Enter to start test sequence...")
        
        # Run all steps
        steps = [
            ("ASCII Parser", self.step1_test_ascii_parser),
            ("Layout Generator", self.step2_test_layout_generator),
            ("Computer Vision + Camera", self.step3_test_cv_tools),
            ("Coordinate Transform", self.step4_coordinate_transform),
            ("WG25 Location", self.step5_find_wg25_and_plot)
        ]
        
        results = []
        for name, test_func in steps:
            input(f"\nPress Enter to run: {name}...")
            result = test_func()
            results.append((name, result))
            
            if not result:
                print(f"\n❌ Test '{name}' failed. Stopping test sequence.")
                break
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        for name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} - {name}")
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        print(f"\nTotal: {passed}/{total} tests completed")
        
        if passed == total:
            print("\n🎉 All tests passed! Alignment system is ready.")
            print("\n📁 Generated files:")
            print("   • test_1_single_block.png")
            print("   • test_2_rotated_layout.png")
            print("   • test_3_camera_detection.png")
            print("   • test_5_final_wg25_location.png")
            print("   • config/test_layout.json")
        
        return passed == total


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive alignment system testing")
    parser.add_argument('--ascii', type=str, default="./AlignmentSystem/ascii_sample.ASC",
                       help='Path to ASCII file')
    parser.add_argument('--rotation', type=float, default=3.0,
                       help='Simulated rotation angle in degrees (default: 3.0)')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4, 5],
                       help='Run specific step only')
    
    args = parser.parse_args()
    
    # Create tester
    tester = AlignmentSystemTester(ascii_file=args.ascii)
    tester.rotation_angle = args.rotation
    
    print("\n" + "="*70)
    print("ALIGNMENT SYSTEM INTERACTIVE TESTER")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  ASCII file: {args.ascii}")
    print(f"  Simulated rotation: {args.rotation}°")
    print(f"  Simulated translation: {tester.translation} nm")
    print(f"  Camera: {tester.camera.pixel_width}x{tester.camera.pixel_height} pixels")
    print(f"  Camera resolution: {tester.camera.nm_per_pixel} nm/pixel")
    print(f"  Camera FOV: {tester.camera.fov_width_nm/1000:.1f} x {tester.camera.fov_height_nm/1000:.1f} µm")
    
    if args.step:
        # Run specific step
        steps = {
            1: ("ASCII Parser", tester.step1_test_ascii_parser),
            2: ("Layout Generator", tester.step2_test_layout_generator),
            3: ("Computer Vision + Camera", tester.step3_test_cv_tools),
            4: ("Coordinate Transform", tester.step4_coordinate_transform),
            5: ("WG25 Location", tester.step5_find_wg25_and_plot)
        }
        
        # Run dependencies first
        if args.step >= 2:
            print("\n🔧 Running dependency: Step 1 (ASCII Parser)")
            tester.step1_test_ascii_parser()
        if args.step >= 3:
            print("\n🔧 Running dependency: Step 2 (Layout Generator)")
            tester.step2_test_layout_generator()
        if args.step >= 4:
            print("\n🔧 Running dependency: Step 3 (Computer Vision + Camera)")
            tester.step3_test_cv_tools()
        if args.step >= 5:
            print("\n🔧 Running dependency: Step 4 (Coordinate Transform)")
            tester.step4_coordinate_transform()
        
        # Run requested step
        name, func = steps[args.step]
        print(f"\n▶️  Running Step {args.step}: {name}")
        success = func()
        
        if success:
            print(f"\n✅ Step {args.step} completed successfully!")
        else:
            print(f"\n❌ Step {args.step} failed!")
    else:
        # Run all tests
        tester.run_all_tests()


if __name__ == "__main__":
    main()