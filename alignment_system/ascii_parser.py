# ascii_parser.py
"""
Parser for EBL ASCII files to extract waveguide and marker positions.
"""
import re
from typing import List, Dict, Tuple, Optional
import numpy as np


class ASCIIParser:
    """Parse EBL ASCII files to extract design geometry."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.polygons = []
        self.text_label = None
        self.text_position = None
        
    def parse(self) -> Dict:
        """Parse ASCII file and return structured data."""
        with open(self.filename, 'r') as f:
            content = f.read()
        
        # Split by '#' to get individual polygons
        blocks = content.split('#')
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
                
            # Check if it's a text block
            if block.startswith('T'):
                self._parse_text(block)
            else:
                self._parse_polygon(block)
        
        # Extract structured data
        return {
            'markers': self._extract_markers(),
            'waveguides': self._extract_waveguides(),
            'gratings': self._extract_gratings(),
            'text_label': self.text_label,
            'text_position': self.text_position
        }
    
    def _parse_polygon(self, block: str):
        """Parse a polygon block."""
        lines = block.strip().split('\n')
        if len(lines) < 2:
            return
        
        # First line: dose, layer
        header = lines[0].strip().split()
        if len(header) < 3:
            return
            
        try:
            layer = int(header[2])
            dose = float(header[1])
        except (ValueError, IndexError):
            return
        
        # Parse coordinates
        coords = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    u = float(parts[0])
                    v = float(parts[1])
                    coords.append((u, v))
                except ValueError:
                    continue
        
        if coords:
            self.polygons.append({
                'layer': layer,
                'dose': dose,
                'coords': coords
            })
    
    def _parse_text(self, block: str):
        """Parse text block."""
        lines = block.strip().split('\n')
        if len(lines) < 4:
            return
        
        # Text position
        pos_line = lines[1].strip().split()
        if len(pos_line) >= 2:
            self.text_position = (float(pos_line[0]), float(pos_line[1]))
        
        # Text content
        if len(lines) >= 4:
            self.text_label = lines[3].strip()
    
    def _extract_markers(self) -> List[Dict]:
        """Extract corner markers (layer 0)."""
        markers = []
        for poly in self.polygons:
            if poly['layer'] == 0:
                coords = poly['coords']
                # Calculate bounding box center
                u_vals = [c[0] for c in coords]
                v_vals = [c[1] for c in coords]
                center_u = (min(u_vals) + max(u_vals)) / 2
                center_v = (min(v_vals) + max(v_vals)) / 2
                
                # Identify corner based on position
                corner = self._identify_corner(center_u, center_v)
                
                markers.append({
                    'corner': corner,
                    'position': (center_u, center_v),
                    'coords': coords
                })
        
        return markers
    
    def _identify_corner(self, u: float, v: float) -> str:
        """Identify which corner a marker is at."""
        # Assume 200x200 block
        u_threshold = 100.0
        v_threshold = 100.0
        
        if v > v_threshold:
            if u < u_threshold:
                return 'top_left'
            else:
                return 'top_right'
        else:
            if u < u_threshold:
                return 'bottom_left'
            else:
                return 'bottom_right'
    
    def _extract_waveguides(self) -> List[Dict]:
        """Extract waveguide positions (layer 1, full length)."""
        waveguides = []
        
        for poly in self.polygons:
            if poly['layer'] != 1:
                continue
            
            coords = poly['coords']
            u_vals = [c[0] for c in coords]
            v_vals = [c[1] for c in coords]
            
            u_min, u_max = min(u_vals), max(u_vals)
            v_min, v_max = min(v_vals), max(v_vals)
            
            # Waveguides span from u≈10 to u≈190
            if u_max - u_min > 100:  # Full length waveguides
                waveguides.append({
                    'v_center': (v_min + v_max) / 2,
                    'v_top': v_max,
                    'v_bottom': v_min,
                    'width': v_max - v_min,
                    'u_start': u_min,
                    'u_end': u_max
                })
        
        # Sort by v_center (top to bottom)
        waveguides.sort(key=lambda x: x['v_center'], reverse=True)
        
        # Add waveguide numbers
        for i, wg in enumerate(waveguides):
            wg['number'] = i + 1
        
        return waveguides
    
    def _extract_gratings(self) -> List[Dict]:
        """Extract grating coupler positions (layer 2)."""
        gratings = []
        
        for poly in self.polygons:
            if poly['layer'] != 2:
                continue
            
            coords = poly['coords']
            u_vals = [c[0] for c in coords]
            v_vals = [c[1] for c in coords]
            
            u_center = (min(u_vals) + max(u_vals)) / 2
            v_center = (min(v_vals) + max(v_vals)) / 2
            
            # Determine if left or right grating
            side = 'left' if u_center < 100 else 'right'
            
            gratings.append({
                'position': (u_center, v_center),
                'v_center': v_center,
                'side': side,
                'coords': coords
            })
        
        return gratings


def find_waveguide_grating(waveguides: List[Dict], gratings: List[Dict], 
                          wg_number: int, side: str = 'left') -> Optional[Tuple[float, float]]:
    """
    Find the grating coupler position for a specific waveguide.
    
    Args:
        waveguides: List of waveguide dicts
        gratings: List of grating dicts
        wg_number: Waveguide number (1-indexed)
        side: 'left' or 'right'
    
    Returns:
        (u, v) position of the grating, or None if not found
    """
    # Find the waveguide
    wg = None
    for w in waveguides:
        if w['number'] == wg_number:
            wg = w
            break
    
    if wg is None:
        return None
    
    # Find gratings on the correct side and closest to this waveguide's v position
    side_gratings = [g for g in gratings if g['side'] == side]
    
    if not side_gratings:
        return None
    
    # Find closest grating by v-coordinate
    v_target = wg['v_center']
    closest = min(side_gratings, key=lambda g: abs(g['v_center'] - v_target))
    
    return closest['position']


# Test/example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ascii_parser.py <ascii_file>")
        sys.exit(1)
    
    parser = ASCIIParser(sys.argv[1])
    data = parser.parse()
    
    print(f"Parsed: {parser.filename}")
    print(f"Text label: {data['text_label']}")
    print(f"Text position: {data['text_position']}")
    print(f"\nMarkers found: {len(data['markers'])}")
    for m in data['markers']:
        print(f"  {m['corner']}: {m['position']}")
    
    print(f"\nWaveguides found: {len(data['waveguides'])}")
    for i, wg in enumerate(data['waveguides'][:5]):  # First 5
        print(f"  WG {wg['number']}: v={wg['v_center']:.3f} µm, width={wg['width']:.3f} µm")
    print(f"  ... (showing first 5 of {len(data['waveguides'])})")
    
    print(f"\nGratings found: {len(data['gratings'])}")
    
    # Find waveguide 25, left grating
    wg25_pos = find_waveguide_grating(data['waveguides'], data['gratings'], 25, 'left')
    if wg25_pos:
        print(f"\nWaveguide #25 LEFT grating at: u={wg25_pos[0]:.3f}, v={wg25_pos[1]:.3f} µm")