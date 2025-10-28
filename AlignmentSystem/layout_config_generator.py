# layout_config_generator.py
"""
Generate JSON layout configuration from ASCII file.
Creates positions for all 20 blocks with fiducials and gratings.
"""
import json
from typing import Dict, List
from AlignmentSystem.ascii_parser import ASCIIParser, find_waveguide_grating


def generate_layout_config(ascii_file: str, output_file: str = "config/sample_layout.json"):
    """
    Generate complete layout configuration from single block ASCII file.
    
    Args:
        ascii_file: Path to ASCII file for one block
        output_file: Output JSON file path
    """
    print(f"Parsing ASCII file: {ascii_file}")
    parser = ASCIIParser(ascii_file)
    data = parser.parse()
    
    # Extract template block data
    markers = {m['corner']: m['position'] for m in data['markers']}
    waveguides = data['waveguides']
    gratings = data['gratings']
    
    print(f"Found {len(waveguides)} waveguides in template block")
    
    # Block layout: 4 rows × 5 columns, row-major
    # Center-to-center spacing: 300 µm
    blocks_per_row = 5
    num_rows = 4
    block_spacing = 300.0  # µm
    
    layout = {
        "design_name": "ArrayD_QuasiPeriodic_20Blocks",
        "units": "micrometers",
        "block_size": 200.0,
        "block_spacing": block_spacing,
        "blocks_per_row": blocks_per_row,
        "num_rows": num_rows,
        "total_blocks": num_rows * blocks_per_row,
        "target_waveguide": 25,
        "target_side": "left",
        "blocks": {}
    }
    
    # Generate all 20 blocks
    block_id = 1
    for row in range(num_rows):
        for col in range(blocks_per_row):
            # Calculate block offset
            u_offset = col * block_spacing
            v_offset = row * block_spacing
            
            print(f"Generating block {block_id} (row {row}, col {col})...")
            
            # Create block entry
            block = {
                "id": block_id,
                "row": row,
                "col": col,
                "offset_u": u_offset,
                "offset_v": v_offset,
                "center": [u_offset + 100.0, v_offset + 100.0],
                "fiducials": {},
                "gratings": {},
                "waveguides": {}
            }
            
            # Add fiducials (offset from template)
            for corner, pos in markers.items():
                block["fiducials"][corner] = [
                    pos[0] + u_offset,
                    pos[1] + v_offset
                ]
            
            # Add waveguide 25 grating positions
            wg25_left = find_waveguide_grating(waveguides, gratings, 25, 'left')
            wg25_right = find_waveguide_grating(waveguides, gratings, 25, 'right')
            
            if wg25_left:
                block["gratings"]["wg25_left"] = [
                    wg25_left[0] + u_offset,
                    wg25_left[1] + v_offset
                ]
            
            if wg25_right:
                block["gratings"]["wg25_right"] = [
                    wg25_right[0] + u_offset,
                    wg25_right[1] + v_offset
                ]
            
            # Add all waveguides for reference
            for wg in waveguides:
                wg_num = wg['number']
                block["waveguides"][f"wg{wg_num}"] = {
                    "v_center": wg['v_center'] + v_offset,
                    "width": wg['width'],
                    "u_start": wg['u_start'] + u_offset,
                    "u_end": wg['u_end'] + u_offset
                }
            
            layout["blocks"][block_id] = block
            block_id += 1
    
    # Add calibration fiducials (furthest corners)
    layout["calibration_fiducials"] = {
        "primary": {
            "block_id": 1,
            "corner": "top_left",
            "position": layout["blocks"][1]["fiducials"]["top_left"]
        },
        "secondary": {
            "block_id": 20,
            "corner": "bottom_right",
            "position": layout["blocks"][20]["fiducials"]["bottom_right"]
        }
    }
    
    # Calculate overall dimensions
    layout["total_dimensions"] = {
        "u_min": 0.0,
        "u_max": (blocks_per_row - 1) * block_spacing + 200.0,
        "v_min": 0.0,
        "v_max": (num_rows - 1) * block_spacing + 200.0
    }
    
    # Save to file
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(layout, f, indent=2)
    
    print(f"\n✅ Layout configuration saved to: {output_file}")
    print(f"   Total blocks: {layout['total_blocks']}")
    print(f"   Overall dimensions: {layout['total_dimensions']['u_max']:.1f} × {layout['total_dimensions']['v_max']:.1f} µm")
    
    return layout


def load_layout_config(config_file: str = "config/sample_layout.json") -> Dict:
    """Load layout configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


# Command-line interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python layout_config_generator.py <ascii_file> [output_json]")
        print("\nExample:")
        print("  python layout_config_generator.py ArrayDquasiperiodic1.ASC")
        sys.exit(1)
    
    ascii_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "config/sample_layout.json"
    
    layout = generate_layout_config(ascii_file, output_file)
    
    # Print summary
    print("\n" + "="*70)
    print("LAYOUT SUMMARY")
    print("="*70)
    print(f"Design: {layout['design_name']}")
    print(f"Blocks: {layout['total_blocks']} ({layout['num_rows']}×{layout['blocks_per_row']})")
    print(f"Block spacing: {layout['block_spacing']} µm")
    print(f"Total area: {layout['total_dimensions']['u_max']:.0f} × {layout['total_dimensions']['v_max']:.0f} µm")
    print(f"\nTarget: Block 10, Waveguide {layout['target_waveguide']}, {layout['target_side']} side")
    
    # Show block 10 info (center block)
    block10 = layout['blocks'][10]
    print(f"\nBlock 10 (center):")
    print(f"  Position: row {block10['row']}, col {block10['col']}")
    print(f"  Center: ({block10['center'][0]:.1f}, {block10['center'][1]:.1f}) µm")
    print(f"  WG25 left grating: ({block10['gratings']['wg25_left'][0]:.1f}, {block10['gratings']['wg25_left'][1]:.1f}) µm")
    
    print("\nCalibration fiducials:")
    print(f"  Primary (Block 1, top-left): {layout['calibration_fiducials']['primary']['position']}")
    print(f"  Secondary (Block 20, bottom-right): {layout['calibration_fiducials']['secondary']['position']}")