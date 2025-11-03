# layout_config_generator_v2.py
"""
Generate JSON layout configuration with proper coordinate conventions.

Convention:
- Block positions: Unrotated design coordinates (u, v in µm)
- Block contents: Local coordinates relative to block bottom-left (u, v in µm)
- Rotation/translation: Applied at runtime during coordinate transformation
"""
import json
from typing import Dict
from pathlib import Path
from AlignmentSystem.ascii_parser import ASCIIParser


def generate_layout_config_v2(
    ascii_file: str,
    output_file: str = "config/mock_layout.json",
    simulated_rotation: float = 3.0,
    simulated_translation: tuple = (10000, 0)
):
    """
    Generate layout configuration with proper coordinate conventions.
    
    Args:
        ascii_file: Path to ASCII file for one block template
        output_file: Output JSON file path
        simulated_rotation: Rotation angle in degrees (for simulation ground truth)
        simulated_translation: (Y, Z) translation in nm (for simulation ground truth)
    
    Returns:
        dict: Layout configuration
    """
    print(f"Parsing ASCII file: {ascii_file}")
    parser = ASCIIParser(ascii_file)
    data = parser.parse()
    
    # Extract template block data
    markers = {m['corner']: m['position'] for m in data['markers']}
    waveguides = data['waveguides']
    gratings = data['gratings']
    
    print(f"Found {len(waveguides)} waveguides in template block")
    
    # Block layout parameters
    blocks_per_row = 5
    num_rows = 4
    block_spacing = 300.0  # µm center-to-center
    block_size = 200.0     # µm
    
    layout = {
        "design_name": "ArrayD_QuasiPeriodic_20Blocks",
        "version": "2.0",
        "coordinate_system": {
            "description": "Block positions in unrotated design coordinates (µm). Contents in local block coordinates (µm).",
            "origin": "Bottom-left corner of Block 1 at stage (0, 0)",
            "units_blocks": "micrometers",
            "units_stage": "nanometers"
        },
        "block_layout": {
            "block_size": block_size,
            "block_spacing": block_spacing,
            "blocks_per_row": blocks_per_row,
            "num_rows": num_rows,
            "total_blocks": num_rows * blocks_per_row
        },
        "simulation_ground_truth": {
            "description": "True rotation/translation for testing (mock mode only)",
            "rotation_deg": simulated_rotation,
            "translation_nm": list(simulated_translation),
            "note": "Real system must determine these by finding fiducial markers"
        },
        "calibration_fiducials": {
            "primary": {"block_id": 1, "corner": "top_left"},
            "secondary": {"block_id": 20, "corner": "bottom_right"}
        },
        "target": {
            "block_id": 10,
            "waveguide": 25,
            "side": "left"
        },
        "blocks": {}
    }
    
    # Generate all 20 blocks
    block_id = 1
    for row in range(num_rows):
        for col in range(blocks_per_row):
            # Calculate block design position (unrotated)
            # Block 1 at (0, 0), spacing between centers
            u_center = col * block_spacing
            v_center = -1 * row * block_spacing
            
            print(f"Generating block {block_id} at design position ({u_center}, {v_center}) µm...")
            
            # Create block entry
            block = {
                "id": block_id,
                "row": row,
                "col": col,
                "design_position": [u_center, v_center],  # Center position, unrotated
                "fiducials": {},
                "waveguides": {},
                "gratings": {}
            }
            
            # Add fiducials in LOCAL coordinates (relative to block bottom-left)
            # Template markers are already in local coords (0-200 µm range)
            for corner, local_pos in markers.items():
                block["fiducials"][corner] = list(local_pos)
            
            # Add waveguides in LOCAL coordinates
            for wg in waveguides:
                wg_id = f"wg{wg['number']}"
                block["waveguides"][wg_id] = {
                    "number": wg['number'],
                    "v_center": wg['v_center'],  # Local v coordinate
                    "width": wg['width'],
                    "u_start": wg['u_start'],    # Local u coordinate
                    "u_end": wg['u_end']
                }
            
            # Add gratings in LOCAL coordinates
            # Store only WG25 gratings for now
            for g in gratings:
                # Find which waveguide this grating belongs to
                v_grating = g['v_center']
                
                # Find closest waveguide
                closest_wg = min(waveguides, key=lambda w: abs(w['v_center'] - v_grating))
                
                if closest_wg['number'] == 25:
                    side = g['side']
                    grating_id = f"wg25_{side}"
                    block["gratings"][grating_id] = {
                        "position": list(g['position']),  # Local (u, v)
                        "side": side,
                        "waveguide": 25
                    }
            
            layout["blocks"][block_id] = block
            block_id += 1
    
    # Save to file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(layout, f, indent=2)
    
    print(f"\n✅ Layout configuration saved to: {output_file}")
    print(f"   Version: 2.0 (local coordinate convention)")
    print(f"   Total blocks: {layout['block_layout']['total_blocks']}")
    print(f"   Simulation rotation: {simulated_rotation}°")
    print(f"   Simulation translation: {simulated_translation} nm")
    
    return layout


def load_layout_config_v2(config_file: str = "config/mock_layout.json") -> Dict:
    """Load layout configuration v2."""
    with open(config_file, 'r') as f:
        layout = json.load(f)
    
    # Convert block keys to integers
    if "blocks" in layout and isinstance(layout["blocks"], dict):
        layout["blocks"] = {int(k): v for k, v in layout["blocks"].items()}
    
    return layout

def plot_layout_v2(layout: Dict, output_path: str = "config/mock_layout.png"):
    """
    Visualize the block layout (design + simulated ground truth) and save as an image,
    automatically labeling corners (top_left, bottom_left, etc.) from layout data.

    Args:
        layout: Layout configuration dictionary.
        output_path: File path to save the PNG image.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from AlignmentSystem.coordinate_utils import CoordinateConverter

    block_size = layout["block_layout"]["block_size"]
    blocks = layout["blocks"]
    rotation_deg = layout["simulation_ground_truth"]["rotation_deg"]
    translation_nm = layout["simulation_ground_truth"]["translation_nm"]
    translation_um = (translation_nm[0] / 1000.0, translation_nm[1] / 1000.0)

    # Initialize converter
    converter = CoordinateConverter(layout)
    converter.set_transformation(rotation_deg, translation_nm)

    plt.figure(figsize=(12, 10))
    plt.title(f"Layout: {layout['design_name']}\n"
              f"Rotation = {rotation_deg:.2f}°, Translation = {translation_um} µm")
    plt.xlabel("u (µm)")
    plt.ylabel("v (µm)")

    corner_colors = {
        "top_left": "orange",
        "top_right": "green",
        "bottom_left": "blue",
        "bottom_right": "red"
    }

    for bid, block in blocks.items():
        u_center, v_center = block["design_position"]

        # --- Design rectangle (unrotated)
        u0, v0 = u_center - block_size / 2, v_center - block_size / 2
        rect_design = plt.Rectangle((u0, v0), block_size, block_size,
                                    edgecolor="navy", facecolor="none", lw=0.8)
        plt.gca().add_patch(rect_design)
        plt.text(u_center, v_center, str(bid),
                 ha="center", va="center", fontsize=7, color="navy")

        # --- Stage rectangle (rotated + translated)
        stage_center_nm = converter.design_to_stage(u_center, v_center)
        stage_center = np.array(stage_center_nm) / 1000.0  # → µm

        u0r, v0r = stage_center - block_size / 2
        rect_stage = plt.Rectangle((u0r, v0r), block_size, block_size,
                                   edgecolor="crimson", facecolor="none", lw=0.8, ls="--")
        plt.gca().add_patch(rect_stage)

        # Line between centers
        plt.plot([u_center, stage_center[0]], [v_center, stage_center[1]],
                 color="gray", lw=0.5, alpha=0.6)

        # --- Fiducials (use layout data)
        for corner, (u_local, v_local) in block["fiducials"].items():
            color = corner_colors.get(corner, "black")

            # Global (design) coordinates
            design_corner = np.array([u0 + u_local, v0 + v_local])

            # Stage coordinates (convert nm → µm)
            stage_corner_nm = converter.block_local_to_stage(bid, u_local, v_local)
            stage_corner = np.array(stage_corner_nm) / 1000.0

            # Plot design corner
            plt.scatter(design_corner[0], design_corner[1], s=12, c=color, marker='o',
                        label=f"{corner} (design)" if bid == 1 else "")
            plt.text(design_corner[0], design_corner[1],
                     corner.replace("_", "\n"), fontsize=6,
                     ha="center", va="center", color=color)

            # Plot stage corner
            plt.scatter(stage_corner[0], stage_corner[1], s=12, c=color, marker='x',
                        label=f"{corner} (stage)" if bid == 1 else "")
            plt.text(stage_corner[0], stage_corner[1],
                     corner.replace("_", "\n"), fontsize=6,
                     ha="center", va="center", color=color)

            # Connect them
            plt.plot([design_corner[0], stage_corner[0]],
                     [design_corner[1], stage_corner[1]],
                     color=color, lw=0.4, alpha=0.5)

    plt.legend(loc="upper right", fontsize=7, frameon=True)
    plt.axis("equal")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"🖼️ Layout image with labeled fiducial corners saved to: {output_path}")

# Test/example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python layout_config_generator_v2.py <ascii_file> [output_json]")
        print("\nExample:")
        print("  python layout_config_generator_v2.py AlignmentSystem/ascii_sample.ASC")
        sys.exit(1)
    
    ascii_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "config/mock_layout.json"
    
    # Generate with simulated rotation/translation
    layout = generate_layout_config_v2(
        ascii_file,
        output_file,
        simulated_rotation=3.0,
        simulated_translation=(10000, 0)
    )
    
    # Print summary
    print("\n" + "="*70)
    print("LAYOUT SUMMARY")
    print("="*70)
    print(f"Design: {layout['design_name']}")
    print(f"Version: {layout['version']}")
    print(f"Blocks: {layout['block_layout']['total_blocks']}")
    print(f"Block spacing: {layout['block_layout']['block_spacing']} µm")
    
    # Show Block 1 info
    block1 = layout['blocks'][1]
    print(f"\nBlock 1:")
    print(f"  Design position: {block1['design_position']} µm")
    print(f"  Fiducials (local coords):")
    for corner, pos in block1['fiducials'].items():
        print(f"    {corner}: {pos} µm")
    
    # Show Block 10 info
    block10 = layout['blocks'][10]
    print(f"\nBlock 10 (target):")
    print(f"  Design position: {block10['design_position']} µm")
    if 'wg25_left' in block10['gratings']:
        grating = block10['gratings']['wg25_left']
        print(f"  WG25 left grating (local): {grating['position']} µm")

    plot_layout_v2(layout, "config/mock_layout.png")