# 🚀 Getting Started with Microscope Alignment GUI

## ✅ What's Been Built

A complete **PyQt6 application skeleton** with:

### Core System ✓
- `app/system_state.py` - Centralized state management (all coordinates in µm)
- `app/signals.py` - Qt signals hub for loose coupling
- `app/main.py` - Application entry point with hardware selection
- `app/main_window.py` - Main window with all panels

### Controllers ✓
- `app/controllers/camera_stream.py` - Camera thread with color scaling
- `app/controllers/hardware_manager.py` - Mock/real hardware detection

### Widgets ✓
- `app/widgets/camera_view.py` - Live camera display
- `app/widgets/stage_control.py` - Stage jog controls
- `app/widgets/block_grid.py` - 5×4 block selector
- `app/widgets/waveguide_panel.py` - Waveguide navigation table
- `app/widgets/alignment_panel.py` - Alignment controls
- `app/widgets/status_bar.py` - Custom status bar

### Features Implemented ✓
1. **Live camera stream** with colormaps (gray/jet/hot/viridis/etc.)
2. **Stage control** with jog buttons and Go To positioning
3. **Block grid** with color-coded status indicators
4. **Waveguide table** with navigation buttons
5. **Hardware selection** (mock/real at startup)
6. **State persistence** (save/load to JSON)
7. **Menu system** with shortcuts

## 📦 Installation

### Step 1: Install Dependencies

```bash
pip install PyQt6 numpy opencv-python matplotlib
```

Or use the requirements file:

```bash
pip install -r requirements_gui.txt
```

### Step 2: Verify File Structure

Make sure your project has this structure:

```
your_project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── main_window.py
│   ├── system_state.py
│   ├── signals.py
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── camera_stream.py
│   │   └── hardware_manager.py
│   └── widgets/
│       ├── __init__.py
│       ├── camera_view.py
│       ├── stage_control.py
│       ├── block_grid.py
│       ├── waveguide_panel.py
│       ├── alignment_panel.py
│       └── status_bar.py
│
├── config/
│   └── mock_layout.json          # Required for mock mode!
│
├── HardwareControl/               # Your existing hardware code
├── AlignmentSystem/               # Your existing alignment code
└── run_gui.py                     # Simple launcher
```

### Step 3: Verify Mock Layout Exists

The mock camera needs `config/mock_layout.json`. If you don't have it, generate it:

```bash
python config/layout_config_generator_v3.py AlignmentSystem/ascii_sample.ASC config/mock_layout.json
```

## 🎮 Running the Application

### Method 1: Direct Launch

```bash
python app/main.py
```

### Method 2: Using Launcher

```bash
python run_gui.py
```

### Method 3: From Project Root

```bash
python -m app.main
```

## 🖥️ First Run

When you start the application:

1. **Hardware Selection Dialog** appears
   - Shows detected hardware (camera/stage)
   - Select "Mock Hardware" (recommended for first run)
   - Click "Continue"

2. **Main Window Opens**
   - Camera live stream starts automatically
   - Stage position displays update
   - All controls are functional

3. **Test Basic Features**
   - Change colormap (gray → jet → hot)
   - Use jog buttons to move stage (mock)
   - Click blocks in the grid
   - View waveguide table

## 🧪 Testing the UI

### Camera Controls
- **Colormap**: Switch between different visualization modes
- **Auto-scale**: Toggle automatic brightness/contrast
- **Manual scale**: Set min/max values manually
- **Zoom**: Change magnification or fit to window

### Stage Control
- **Jog buttons**: Move stage in X, Y, Z directions
- **Step size**: Change movement increment (0.1 µm to 500 µm)
- **Go To**: Move to absolute positions
- **Position display**: Updates in real-time (mock: immediate, real: ~10 Hz)

### Block Selection
- **Click any block** in the 5×4 grid
- Selected block highlighted with blue border
- Waveguide table updates automatically
- Info label shows block status

### Waveguide Navigation
- **Select target WG**: Use spinner (1-50)
- **Choose side**: Left/Center/Right grating
- **Go to Target**: Navigate to selected position

## ⚠️ Known Limitations (Phase 1 Skeleton)

These features have **UI ready** but **no implementation yet**:

- ❌ Global alignment algorithm
- ❌ Block calibration algorithm
- ❌ Autofocus
- ❌ Actual coordinate predictions (navigation shows stub message)
- ❌ Progress dialogs for long operations
- ❌ Export functions

**Next Phase**: Implement controllers for these features.

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'app'"

**Solution**: Make sure you're running from project root, or add to path:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python app/main.py
```

### "FileNotFoundError: config/mock_layout.json"

**Solution**: Generate the layout file:

```bash
python config/layout_config_generator_v3.py AlignmentSystem/ascii_sample.ASC config/mock_layout.json
```

### "No module named 'PyQt6'"

**Solution**: Install PyQt6:

```bash
pip install PyQt6
```

### Camera Stream Shows Black Screen

**Possible causes**:
1. **Mock mode**: Check that `MockCamera` can load layout
2. **Real mode**: Verify camera drivers installed
3. Look for error messages in terminal

**Debug**: Add print statements in `MockCamera.acquire_single_image()`

### Stage Buttons Don't Work

**Check**:
1. Hardware connected (green icons in status bar)
2. Look for exceptions in terminal
3. Try smaller step sizes first (0.1 µm)

## 🎯 Next Steps

Now that the skeleton is working, you can:

### Immediate Next Steps

1. **Test Mock Mode**
   ```bash
   python app/main.py
   # Select "Mock Hardware"
   # Verify camera stream works
   # Test stage jog controls
   # Select blocks and view waveguides
   ```

2. **Integrate Alignment System**
   - Import `AlignmentSearcher` from your existing code
   - Create `app/controllers/alignment_controller.py`
   - Implement `_run_global` in `alignment_panel.py`
   - Add progress dialog

3. **Add Navigation**
   - Import `CoordinateTransformV3`
   - Implement `_goto_waveguide` in `waveguide_panel.py`
   - Calculate positions using transform
   - Move stage to predicted coordinates

4. **Implement Autofocus**
   - Create `app/controllers/autofocus_controller.py`
   - Add focus metric calculation
   - Show live plot during scan

### Customization

Edit `system_state.py` to adjust:
- Default camera settings (colormap, scaling)
- Stage jog step sizes
- Alignment parameters
- Window sizes

Edit `main_window.py` to customize:
- Menu structure
- Panel layout
- Keyboard shortcuts

## 📚 Documentation

- **README_GUI.md**: Full feature list and architecture
- **docs/DEVELOPER_GUIDE.md**: Implementation guide
- **Code comments**: Extensive docstrings in all files

## 🐛 Reporting Issues

If you encounter problems:

1. Check terminal output for errors
2. Verify all dependencies installed
3. Try with mock hardware first
4. Check file paths are correct

## 🎉 Success Indicators

You'll know it's working when you can:

- ✓ Launch application without errors
- ✓ See hardware selection dialog
- ✓ Watch live camera stream in main window
- ✓ Use jog buttons to move stage (position updates)
- ✓ Click blocks and see selection change
- ✓ View waveguide table for selected block
- ✓ Change colormaps and see visual updates
- ✓ Save/load state via File menu

---

**Ready to start? Run:**

```bash
python app/main.py
```

**Enjoy your new microscope control interface! 🔬**