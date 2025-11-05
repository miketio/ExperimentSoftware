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
import numpy as np
from AlignmentSystem.ascii_parser import ASCIIParser


def generate_layout_config_v2(
    ascii_file: str,
    output_file: str = "config/mock_layout.json",
    simulated_rotation: float = 0.0,
    simulated_translation: tuple = (0.0, 0.0),  # NOW IN µm (was nm)
    block_rotation_std: float = 10.0,  # NEW: std dev for block angles (degrees)
    block_translation_std: float = 0.0,  # NEW: std dev for block shifts (µm)
    random_seed: int = 42  # NEW: for deterministic errors
):
    """
    Generate layout configuration with proper coordinate conventions.

    Args:
        ascii_file: Path to ASCII file for one block template
        output_file: Output JSON file path
        simulated_rotation: Global sample rotation angle in degrees
        simulated_translation: Global (Y, Z) translation in µm (NOT nm)
        block_rotation_std: Std deviation for per-block rotation errors (degrees)
        block_translation_std: Std deviation for per-block translation errors (µm)
        random_seed: Random seed for deterministic fabrication errors

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
        "version": "2.1",  # Version bump
        "coordinate_system": {
            "description": "All coordinates in micrometers (µm). Stage hardware uses nm internally.",
            "origin": "Bottom-left corner of Block 1 at stage (0, 0)",
            "units": "micrometers"
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
            "translation_um": list(simulated_translation),  # NOW IN µm
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
    
    # Set random seed for deterministic errors
    np.random.seed(random_seed)
    # Generate all 20 blocks
    block_id = 1
    for row in range(num_rows):
        for col in range(blocks_per_row):
            # Calculate block design position (unrotated)
            # Block 1 at (0, 0), spacing between centers
            u_center = col * block_spacing
            v_center = -1 * row * block_spacing
            
            print(f"Generating block {block_id} at design position ({u_center}, {v_center}) µm...")
            
            # Generate deterministic fabrication errors for this block
            fab_rotation = np.random.normal(0, block_rotation_std)
            fab_translation_Y = np.random.normal(0, block_translation_std)
            fab_translation_Z = np.random.normal(0, block_translation_std)
            
            # Create block entry
            block = {
                "id": block_id,
                "row": row,
                "col": col,
                "design_position": [u_center, v_center],  # µm, unrotated
                "fabrication_error": {  # NEW
                    "rotation_deg": fab_rotation,
                    "translation_um": [fab_translation_Y, fab_translation_Z]
                },
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
    print(f"   Simulation translation: {simulated_translation} µm")
    
    return layout


def load_layout_config_v2(config_file: str = "config/mock_layout.json") -> Dict:
    """Load layout configuration v2."""
    with open(config_file, 'r') as f:
        layout = json.load(f)
    
    # Convert block keys to integers
    if "blocks" in layout and isinstance(layout["blocks"], dict):
        layout["blocks"] = {int(k): v for k, v in layout["blocks"].items()}
    
    return layout

def plot_layout_v2(layout: dict, output_path: str = "config/mock_layout.png"):
    """
    Visualize the block layout (design + fabricated + stage) and save as an image,
    showing the effects of both per-block fabrication errors and global sample transform.

    Args:
        layout: Layout configuration dictionary.
        output_path: File path to save the PNG image.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    # We must import the transform class to use its methods
    from AlignmentSystem.coordinate_transform import CoordinateTransform

    block_size = layout["block_layout"]["block_size"]
    blocks = layout["blocks"]
    rotation_deg = layout["simulation_ground_truth"]["rotation_deg"]
    translation_um = layout["simulation_ground_truth"]["translation_um"]
    
    # Initialize converter
    converter = CoordinateTransform(layout)
    converter.set_transformation(rotation_deg, tuple(translation_um))

    plt.figure(figsize=(14, 12))
    plt.title(f"Layout: {layout['design_name']}\n"
              f"Global Rotation = {rotation_deg:.2f}°, Global Translation = {translation_um} µm",
              fontsize=14)
    plt.xlabel("Stage Y / Design u (µm)", fontsize=12)
    plt.ylabel("Stage Z / Design v (µm)", fontsize=12)

    # Define ideal local corners: [bottom_left, bottom_right, top_right, top_left]
    # We use this as the base for all transformations.
    local_corners = np.array([
        [0, 0],
        [block_size, 0],
        [block_size, block_size],
        [0, block_size]
    ])

    # For manual legend entries
    legend_handles = [
        plt.Line2D([0], [0], color='gray', lw=1, label='Ideal Design'),
        plt.Line2D([0], [0], color='blue', lw=1.5, ls='--', label='Fabricated (Design + Fab Error)'),
        plt.Line2D([0], [0], color='red', lw=1.5, ls='--', label='Actual Stage (Fab Error + Global Transform)')
    ]

    for bid, block in blocks.items():
        u_center, v_center = block["design_position"]
        # Ideal bottom-left corner in global design coordinates
        u_bl = u_center - block_size / 2.0
        v_bl = v_center - block_size / 2.0

        # --- 1. Ideal Design Rectangle (Gray Box) ---
        rect_design = plt.Rectangle((u_bl, v_bl), block_size, block_size,
                                    edgecolor="gray", facecolor="none", lw=1.0)
        plt.gca().add_patch(rect_design)
        plt.text(u_center, v_center, str(bid),
                 ha="center", va="center", fontsize=8, color="black")

        # --- 2. Fabricated Design Rectangle (Blue Dashed Polygon) ---
        # This shows where the block *actually* is in the design, including
        # its unique fabrication error, but *before* global sample transform.
        fab_corners_global = []
        for u_local, v_local in local_corners:
            # Apply per-block fab error
            u_fab, v_fab = converter._apply_block_fabrication_error(bid, u_local, v_local)
            
            # Add the block's global design position (bottom-left)
            u_global_fab = u_bl + u_fab
            v_global_fab = v_bl + v_fab
            fab_corners_global.append([u_global_fab, v_global_fab])
        
        poly_fab = plt.Polygon(fab_corners_global, closed=True,
                               edgecolor="blue", facecolor="none", lw=1.5, ls="--")
        plt.gca().add_patch(poly_fab)

        # --- 3. Actual Stage Rectangle (Red Dashed Polygon) ---
        # This shows where the block *ends up* on the stage after
        # *both* fabrication error and global sample transform are applied.
        stage_corners = []
        for u_local, v_local in local_corners:
            # This function handles BOTH fab error AND global transform
            Y, Z = converter.block_local_to_stage(bid, u_local, v_local)
            stage_corners.append([Y, Z])
        
        poly_stage = plt.Polygon(stage_corners, closed=True,
                                 edgecolor="red", facecolor="none", lw=1.5, ls="--")
        plt.gca().add_patch(poly_stage)


        # --- 4. Fiducials (Unchanged) ---
        # This code remains correct. It plots the "fabricated" design fiducial
        # and the "actual" stage fiducial, which should line up
        # perfectly with the corners of our new polygons.
        for corner, (u_local, v_local) in block["fiducials"].items():
            
            # Get "fabricated" design position
            u_fab, v_fab = converter._apply_block_fabrication_error(bid, u_local, v_local)
            design_corner = np.array([u_bl + u_fab, v_bl + v_fab])

            # Get "actual" stage position
            stage_corner = converter.block_local_to_stage(bid, u_local, v_local)
            
            # Plot design corner (blue circle)
            plt.scatter(design_corner[0], design_corner[1], s=15, c='blue', marker='o',
                        label=f"{corner} (fab)" if bid == 1 else "")
            
            # Plot stage corner (red X)
            plt.scatter(stage_corner[0], stage_corner[1], s=20, c='red', marker='x',
                        label=f"{corner} (stage)" if bid == 1 else "")

            # Connect them
            plt.plot([design_corner[0], stage_corner[0]],
                     [design_corner[1], stage_corner[1]],
                     color="orange", lw=0.5, alpha=0.7)

    # Add the custom polygon handles to the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=legend_handles + handles, loc="upper right", fontsize=8, frameon=True)
    
    plt.axis("equal")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for title

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"🖼️ Layout image with fabricated errors saved to: {output_path}")
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
        simulated_translation=(0.0, 0.0),  # µm now
        block_rotation_std=10.0,  # ~1° per block
        block_translation_std=1.0  # ~1 µm per block
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