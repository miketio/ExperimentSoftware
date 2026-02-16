#!/usr/bin/env python3
"""
K-Space Reconstruction from Filter Sweep - Full Version with Log Analysis

Features:
- "Find and Place" reconstruction logic
- Maps physical filter position to k-space X-coordinate
- Detects vertical slit position in source images
- Interpolates to fixed 2048x2048 output grid
- Includes detailed Logarithmic Scale Analysis

Date: 2026-01-20
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import tifffile
from pathlib import Path
from typing import Dict, List, Optional
import h5py
from scipy.interpolate import interp1d
import sys
import traceback

class KSpaceAnalyzer:
    """
    K-space reconstruction from filter sweep data.
    
    Logic:
    1. Finds the bright vertical line in the source image.
    2. Maps the filter's physical position (µm) to a column index in the 2048x2048 output.
    3. Places the extracted column into the calculated position.
    """
    
    def __init__(self, sweep_dir: str):
        self.sweep_dir = Path(sweep_dir)
        self.metadata = None
        self.images = []
        self.positions_um = []
        self.kspace_2d = None
        
    def load_metadata(self) -> Dict:
        """Load and validate sweep metadata."""
        metadata_file = self.sweep_dir / "sweep_metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        print("="*70)
        print("SWEEP METADATA")
        print("="*70)
        print(f"📂 Sweep directory: {self.sweep_dir}")
        
        config = self.metadata['sweep_config']
        print(f"📏 Sweep range: {config['start_um']:.1f} to {config['end_um']:.1f} µm")
        print(f"📊 Total positions: {config['total_positions']}")
        
        return self.metadata
    
    def load_images(self, max_images: Optional[int] = None) -> List[np.ndarray]:
        """Load all sweep images with correct path handling."""
        print("\nLoading images...")
        
        results = self.metadata['results']
        if max_images:
            results = results[:max_images]
        
        self.images = []
        self.positions_um = []
        
        for i, result in enumerate(results):
            if 'error' in result or result['image_file'] is None:
                continue
            
            img_filename = Path(result['image_file']).name
            img_path = self.sweep_dir / img_filename
            
            if not img_path.exists():
                continue
            
            try:
                img = tifffile.imread(str(img_path))
                self.images.append(img)
                self.positions_um.append(result['actual_nm'] / 1000.0)
                
                if i % 50 == 0:
                    print(f"  Loaded {i+1}/{len(results)}...")
            
            except Exception as e:
                print(f"  ⚠️  Failed to load {img_filename}: {e}")
        
        print(f"✅ Loaded {len(self.images)} images")
        return self.images

    def find_bright_column(self, img: np.ndarray) -> int:
        """Find the X-position of the brightest vertical column."""
        column_sums = np.sum(img, axis=0)
        brightest_x = np.argmax(column_sums)
        return brightest_x

    def reconstruct_kspace_correct(
        self, 
        output_size: int = 2048,
        search_mode: str = 'brightest',  # 'brightest' or 'metadata'
        filter_range_um: tuple = (-10500, 500)  # From metadata
    ) -> np.ndarray:
        """
        Reconstruct 2D k-space by placing each vertical slice at its correct position.
        """
        if not self.images:
            raise RuntimeError("No images loaded. Call load_images() first.")
        
        print(f"\n🔧 Reconstructing k-space (CORRECTED METHOD)...")
        print(f"   Output size: {output_size} × {output_size}")
        print(f"   Search mode: {search_mode}")
        print(f"   Filter Mapping: {filter_range_um[0]}µm (Left) -> {filter_range_um[1]}µm (Right)")
        
        h, w = self.images[0].shape
        n_positions = len(self.images)
        
        # Initialize k-space
        self.kspace_2d = np.zeros((output_size, output_size), dtype=np.float32)
        
        min_filter_um, max_filter_um = filter_range_um
        
        for i, img in enumerate(self.images):
            filter_pos_um = self.positions_um[i]
            
            # === STEP 1: Find where the bright line is in THIS image ===
            if search_mode == 'brightest':
                source_x = self.find_bright_column(img)
                
            elif search_mode == 'metadata':
                # Use filter position to estimate where line should be
                norm_pos = (filter_pos_um - min_filter_um) / (max_filter_um - min_filter_um)
                source_x = int(norm_pos * w)
                source_x = np.clip(source_x, 0, w - 1)
            else:
                raise ValueError(f"Unknown search mode: {search_mode}")
            
            # === STEP 2: Extract the vertical column from source image ===
            column = img[:, source_x]  # All Y values at this X position
            
            # === STEP 3: Determine where to place it in k-space ===
            normalized_pos = (filter_pos_um - min_filter_um) / (max_filter_um - min_filter_um)
            target_x = int(normalized_pos * (output_size - 1))
            target_x = np.clip(target_x, 0, output_size - 1)
            
            # === STEP 4: Place column in k-space (with interpolation if needed) ===
            if len(column) == output_size:
                self.kspace_2d[:, target_x] = column
            else:
                # Resize column to match output size
                y_old = np.linspace(0, 1, len(column))
                y_new = np.linspace(0, 1, output_size)
                interpolator = interp1d(y_old, column, kind='linear', fill_value='extrapolate')
                self.kspace_2d[:, target_x] = interpolator(y_new)
            
            if i % 50 == 0:
                print(f"  [{i+1}/{n_positions}] Filter: {filter_pos_um:+7.1f} µm → "
                      f"Source X: {source_x:4d} → Target X: {target_x:4d}")
        
        print(f"\n✅ K-space reconstructed!")
        print(f"   Shape: {self.kspace_2d.shape}")
        
        return self.kspace_2d

    def diagnose_reconstruction_quality(self):
        """Diagnose the quality of k-space reconstruction."""
        if self.kspace_2d is None:
            raise RuntimeError("K-space not reconstructed")
        
        print("\n" + "="*70)
        print("🔍 K-SPACE RECONSTRUCTION DIAGNOSTICS")
        print("="*70)
        
        h, w = self.kspace_2d.shape
        column_sums = np.sum(self.kspace_2d, axis=0)
        filled_columns = np.count_nonzero(column_sums)
        
        print(f"📊 Filled columns: {filled_columns}/{w} ({filled_columns/w*100:.1f}%)")
        
        nonzero_vals = self.kspace_2d[self.kspace_2d > 0]
        if len(nonzero_vals) > 0:
            print(f"📊 Intensity (non-zero pixels):")
            print(f"   Mean: {np.mean(nonzero_vals):.1f}")
            print(f"   Max:  {np.max(nonzero_vals):.1f}")
            
        print("="*70 + "\n")
    
    def save_kspace(self, output_prefix: str = "kspace_full_2048x2048"):
        """Save reconstructed k-space to disk."""
        output_dir = self.sweep_dir
        
        # Save as NPY
        npy_file = output_dir / f"{output_prefix}.npy"
        np.save(npy_file, self.kspace_2d)
        print(f"💾 Saved k-space as NPY: {npy_file}")
        
        # Save as HDF5
        h5_file = output_dir / f"{output_prefix}.h5"
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('kspace_2d', data=self.kspace_2d)
            f.attrs['sweep_dir'] = str(self.sweep_dir)
            f.attrs['positions_um'] = self.positions_um
        print(f"💾 Saved k-space as HDF5: {h5_file}")

    def plot_reconstruction_summary(self):
        """Standard Linear Scale Plot."""
        if self.kspace_2d is None: return
        
        print("\n📊 Generating linear summary plot...")
        plt.style.use('dark_background')
        
        fig = plt.figure(figsize=(20, 12), facecolor='black')
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Full k-space
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        kspace_norm = self.kspace_2d / np.max(self.kspace_2d)
        im1 = ax1.imshow(kspace_norm, aspect='equal', cmap='hot', origin='lower', vmin=0, vmax=0.3)
        ax1.set_title('Full K-Space (Linear Scale)', color='yellow')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).set_label('Intensity')
        
        # 2. Zoomed view
        ax2 = fig.add_subplot(gs[0, 2])
        column_sums = np.sum(self.kspace_2d, axis=0)
        filled_cols = np.where(column_sums > 0)[0]
        if len(filled_cols) > 0:
            x_min = max(0, filled_cols[0] - 50)
            x_max = min(2047, filled_cols[-1] + 50)
            ax2.imshow(kspace_norm[:, x_min:x_max], aspect='auto', cmap='hot', origin='lower', vmin=0, vmax=0.3)
            ax2.set_title(f'Zoomed (X: {x_min}-{x_max})', color='yellow')
        
        # 3. Column occupation
        ax3 = fig.add_subplot(gs[1, 2])
        occupation = (column_sums > 0).astype(int)
        ax3.plot(occupation, color='lime', linewidth=1)
        ax3.fill_between(range(len(occupation)), occupation, alpha=0.3, color='lime')
        ax3.set_title('Column Occupation', color='yellow')
        
        # 4. Intensity Profile
        ax4 = fig.add_subplot(gs[2, :])
        ax4.plot(column_sums, color='cyan', linewidth=1)
        ax4.set_title('Intensity Profile (Summed Y)', color='yellow')
        ax4.set_xlim(0, 2048)

        output_file = self.sweep_dir / "kspace_reconstruction_full.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"📊 Saved Linear Plot: {output_file}")
        plt.show()
        plt.style.use('default')

    def plot_kspace_log_scales(self):
        """
        Generate a 2x2 grid of the reconstructed k-space using different 
        logarithmic scales to reveal low-intensity details.
        """
        if self.kspace_2d is None:
            raise RuntimeError("K-space not reconstructed. Run reconstruction first.")

        print("\n📊 Generating Log-Scale Comparison Plot...")

        # Normalize data so max is 1.0 (10^0)
        max_val = np.max(self.kspace_2d)
        if max_val == 0: 
            print("⚠️ Warning: K-space is empty.")
            return
            
        norm_data = self.kspace_2d / max_val
        # Epsilon to avoid log(0) errors
        eps = 1e-9
        norm_data = np.clip(norm_data, eps, 1.0)

        # Define the exponents for the four plots
        # -1.5 -> ~3.2%
        # -2.0 -> 1.0%
        # -2.5 -> ~0.3%
        # -3.0 -> 0.1%
        log_ranges = [-1.5, -2.0, -2.5, -3.0]
        
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor='black')
        axes = axes.flatten()

        for i, exponent in enumerate(log_ranges):
            ax = axes[i]
            vmin = 10**exponent
            
            # Use Magma for high dynamic range perceptibility
            im = ax.imshow(
                norm_data,
                norm=mcolors.LogNorm(vmin=vmin, vmax=1.0),
                cmap='magma',
                origin='lower',
                aspect='equal'
            )
            
            # Formatting
            percent = vmin * 100
            ax.set_title(f"Log Scale: 10⁰ to 10^{exponent}\n(Shows down to {percent:.2f}% of max)", 
                         color='yellow', fontsize=12, fontweight='bold')
            
            ax.set_xlabel("k_x (pixels)", color='white')
            ax.set_ylabel("k_y (pixels)", color='white')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors='white')
            cbar.set_label('Normalized Intensity (Log)', color='white')

        plt.suptitle(f"Logarithmic Intensity Analysis: {self.sweep_dir.name}", 
                     fontsize=18, color='cyan', y=0.96, fontweight='bold')
        
        output_file = self.sweep_dir / "kspace_log_scales.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='black')
        print(f"📊 Saved Log Plot: {output_file}")
        plt.show()
        plt.style.use('default')

    def plot_debug_grid(self, n_samples: int = 9):
        """Show grid of sample images to visually verify sweep motion."""
        if not self.images: return
        
        print("\n📊 Generating debug grid...")
        n_images = len(self.images)
        sample_indices = np.linspace(0, n_images - 1, n_samples, dtype=int)
        
        rows = int(np.sqrt(n_samples))
        cols = int(np.ceil(n_samples / rows))
        
        plt.style.use('dark_background')
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12), facecolor='black')
        axes = axes.flatten()
        
        for i, idx in enumerate(sample_indices):
            ax = axes[i]
            img = self.images[idx]
            pos_um = self.positions_um[idx]
            bright_x = self.find_bright_column(img)
            
            ax.imshow(img, cmap='hot', origin='lower')
            ax.set_title(f"Pos: {pos_um:.1f} µm\nBright X: {bright_x}", fontsize=10, color='yellow')
            ax.axis('off')
            ax.axvline(bright_x, color='cyan', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        output_file = self.sweep_dir / "sweep_debug_grid.png"
        plt.savefig(output_file, dpi=150, facecolor='black')
        print(f"📊 Saved Debug Grid: {output_file}")
        plt.show()
        plt.style.use('default')


def main():
    if len(sys.argv) < 2:
        print("Usage: python reconstruct_kspace_log.py <sweep_directory>")
        sys.exit(1)
    
    sweep_dir = sys.argv[1]
    
    print("\n" + "="*70)
    print("K-SPACE RECONSTRUCTION (FULL + LOG ANALYSIS)")
    print("="*70 + "\n")
    
    try:
        analyzer = KSpaceAnalyzer(sweep_dir)
        
        # 1. Load Data
        analyzer.load_metadata()
        analyzer.load_images()
        
        # 2. Reconstruct (Correct Method)
        analyzer.reconstruct_kspace_correct(
            output_size=2048,
            search_mode='brightest',  # 'brightest' or 'metadata'
            filter_range_um=(-10500, 500)
        )
        
        # 3. Diagnostics & Saving
        analyzer.diagnose_reconstruction_quality()
        analyzer.save_kspace(output_prefix="kspace_full_2048x2048")
        
        # 4. Visualizations
        analyzer.plot_reconstruction_summary()  # Linear
        analyzer.plot_kspace_log_scales()       # Logarithmic (NEW)
        analyzer.plot_debug_grid()              # Debug checks
        
        print("\n✅ RECONSTRUCTION COMPLETE!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()