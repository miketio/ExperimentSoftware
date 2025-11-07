User Interface Design
3.1 Main Window Layout
┌──────────────────────────────────────────────────────────────────────┐
│ Menu: File | Calibration | Navigation | View | Tools | Help          │
├────────────────────────────────┬─────────────────────────────────────┤
│                                │  ┌─ Stage Control ─────────────────┐│
│                                │  │ Position (µm):                  ││
│    Camera Live View            │  │ X: [_____]  Y: [_____]  Z: [___]││
│    (2048×2048 or scaled)       │  │                                 ││
│                                │  │ Step Size: [10µm ▼]             ││
│    [Crosshair overlay]         │  │                                 ││
│    [Scale bar]                 │  │      [↑]                        ││
│    [FOV indicator]             │  │   [←][●][→]  [Go To]            ││
│                                │  │      [↓]                        ││
│                                │  │                                 ││
│    Controls:                   │  │ [Autofocus] Focus: 0.423        ││
│    Zoom: [100% ▼]              │  └─────────────────────────────────┘│
│    Colormap: [Gray ▼]          │                                     │
│    Min: [50___] Max: [4095___] │  ┌─ Alignment Status ─────────────┐│
│    Auto-scale: [✓]             │  │ Global:  ⬤ Calibrated           ││
│                                │  │   Rot: 3.2°  Trans: (45,32)µm   ││
│                                │  │   Error: 0.08µm                 ││
│                                │  │                                 ││
│                                │  │ Block:   ⬤ Block 10 Active      ││
│                                │  │   Error: 0.05µm                 ││
│                                │  │   [Run Global]  [Calibrate]     ││
│                                │  └─────────────────────────────────┘│
├────────────────────────────────┴─────────────────────────────────────┤
│  ┌─ Block Selection Grid ──────────────────────────────────────────┐ │
│  │                                                                  │ │
│  │   [1]  [2]  [3]  [4]  [5]      Legend:                         │ │
│  │   [6]  [7]  [8]  [9]  [10]     ⬜ Not calibrated                │ │
│  │   [11] [12] [13] [14] [15]     🟨 Global only                   │ │
│  │   [16] [17] [18] [19] [20]     🟩 Block calibrated              │ │
│  │                                 🟦 Currently selected            │ │
│  │   Selected: Block 10                                            │ │
│  │   Position: (0.0, -300.0) µm                                   │ │
│  │   [View All] [Go to Block]                                     │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌─ Waveguide Navigation (Block 10) ──────────────────────────────┐  │
│  │ Filter: [All WGs ▼]  Target: WG25                               │  │
│  │                                                                  │  │
│  │ WG#  │  Position      │  Left Grating  │  Center  │ Right       │  │
│  │ ─────┼────────────────┼────────────────┼──────────┼─────────    │  │
│  │  1   │  (12.5, 175.0) │  [Go] [Info]   │  [Go]    │ [Go]        │  │
│  │  2   │  (12.5, 172.5) │  [Go] [Info]   │  [Go]    │ [Go]        │  │
│  │  ... │                │                │          │             │  │
│  │  25  │  (12.5, 115.0) │  [Go] [Info] ← Target    │ [Go]        │  │
│  │  ... │                │                │          │             │  │
│  │  50  │  (12.5, 12.5)  │  [Go] [Info]   │  [Go]    │ [Go]        │  │
│  │                                                                  │  │
│  │  [Go to Target]  [Set as Target]  [Export Positions]           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  Status Bar: Ready | Stage: X=1234 Y=5678 Z=90 µm | Alignment: OK    │
└────────────────────────────────────────────────────────────────────────┘
3.2 UI Component Specifications
3.2.1 Camera Live View Widget
Purpose: Real-time display of camera frames with overlays and controls
Features:

Live streaming at 10-30 fps (adjustable)
Crosshair overlay showing image center
Scale bar (shows 50µm or 100µm reference)
FOV indicator (shows position in block)
Click-to-move functionality (optional)

Color Scale Controls:

Colormap selection: Grayscale, Jet, Hot, Viridis, etc.
Manual scaling:

Min intensity slider/input (0-65535 for 16-bit)
Max intensity slider/input
Real-time histogram display


Auto-scale modes:

Full range (0-max)
Percentile-based (1%-99%)
Adaptive (running statistics)


Presets: Save/load color scale configurations

Zoom Controls:

Dropdown: 25%, 50%, 100%, 200%, 400%
Fit to window
1:1 pixel mapping

Performance Optimization:

Downsample for display if image > window size
Skip frames if processing falls behind
Adjustable frame rate to reduce CPU load

3.2.2 Stage Control Panel
Purpose: Manual and automated stage positioning
Components:
Position Display:

Live X, Y, Z coordinates in µm
Update rate: 10 Hz
Copy position button (to clipboard)

Manual Control:

Arrow buttons for incremental moves
Step size selector: 0.1µm, 1µm, 10µm, 50µm, 100µm, 500µm
Keyboard shortcuts: Arrow keys, Page Up/Down for Z

Go To Position:

Text inputs for X, Y, Z coordinates
"Go" button to move
Position history dropdown

Safety Features:

Position limits display
Confirmation for large moves (>1000µm)
Emergency stop button

3.2.3 Alignment Control Panel
Purpose: Manage global and block-level calibration
Global Alignment:

"Run Global Alignment" button

Opens progress dialog with live search visualization
Shows current block being searched
Estimated time remaining
Cancel button


Status indicator:

⚪ Not Calibrated
🟡 Calibrating...
🟢 Calibrated
🔴 Failed


Calibration parameters display:

Rotation angle (degrees)
Translation (Y, Z in µm)
Mean/Max error (µm)
Number of fiducials found



Block Alignment:

"Calibrate Selected Block" button
"Calibrate All Blocks" button (batch mode)
Progress indicator for batch operations
Block-specific calibration results

Configuration Button:

Opens settings dialog for alignment parameters:

Search radius (µm)
Step size (µm)
Detection confidence threshold
Enable/disable visualization



3.2.4 Block Selection Grid
Purpose: Visual overview and selection of sample blocks
Layout:

5×4 grid of buttons representing blocks 1-20
Color-coded status:

⬜ Gray: Not calibrated (global transform only)
🟨 Yellow: Globally aligned
🟩 Green: Block calibrated
🟦 Blue: Currently selected
🔴 Red: Alignment failed



Interaction:

Click block to select
Right-click for context menu:

"Go to Block Center"
"Calibrate This Block"
"View Alignment Details"
"Clear Block Calibration"



Info Display:

Selected block number
Design position (u, v)
Predicted stage position (Y, Z)
Calibration status

Additional Features:

"View All" button: Opens detailed grid view with all block info
Search/filter by block number
Highlight blocks with issues

3.2.5 Waveguide Navigation Panel
Purpose: Navigate to specific waveguides and gratings within a block
Table View:

Columns:

WG# (1-50)
Position (local u, v coordinates)
Left Grating [Go] [Info]
Center [Go]
Right Grating [Go] [Info]


Sortable by column
Filterable by waveguide number range

Target Selection:

Radio buttons or highlighting to mark target waveguide
"Go to Target" button (large, prominent)
"Set as Target" button (marks currently displayed WG)
Target indicator (arrow or highlighting in table)

Grating Details:

"Info" button opens popup with:

Predicted position (stage coordinates)
Local position (block coordinates)
Distance from current position
Expected in-focus position (if autofocus history available)



Batch Operations:

"Export Positions" → CSV file with all positions
"Visit All Left Gratings" → automated scan sequence

3.2.6 Status Bar
Purpose: System-wide status information
Sections:

Left: Current operation/message

"Ready"
"Moving to position..."
"Running autofocus..."
"Alignment in progress..."


Center: Stage position (always visible)

"X=1234.5 Y=5678.9 Z=90.0 µm"


Right: System status indicators

Camera: 🟢 Connected / 🔴 Disconnected
Stage: 🟢 Connected / 🔴 Disconnected
Alignment: 🟢 Calibrated / ⚪ Not Calibrated