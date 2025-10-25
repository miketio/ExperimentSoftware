# FILE: stage_commands.py
"""
StageCommandProcessor: a separate module that handles all stage/CLI commands.

Usage:
    from stage_commands import StageCommandProcessor
    processor = StageCommandProcessor(stage_app, camera_app, autofocus)
    processor.process(command)

Supported new ROI commands:
    roi left=400 top=1200 width=1300 height=1000
    setroi 400 1200 1300 1000
    showroi
    roi reset

It preserves the other commands from the original process_stage_command and centralizes parsing and error handling.
"""

import re


class StageCommandProcessor:
    def __init__(self, stage_app, camera_app, autofocus):
        self.stage_app = stage_app
        self.camera_app = camera_app
        self.autofocus = autofocus

        # remember last ROI set via CLI (left, top, width, height)
        self.last_roi = None

    # ---------------------- helpers ----------------------
    def _parse_kv_parts(self, parts):
        """Parse key=value parts into dict. Returns {} if none."""
        d = {}
        for p in parts:
            if '=' in p:
                k, v = p.split('=', 1)
                d[k.strip().lower()] = v.strip()
        return d

    def _try_int(self, s, default=None):
        try:
            return int(s)
        except Exception:
            return default

    # ---------------------- ROI handling ----------------------
    def _handle_roi(self, parts):
        """Handle roi / setroi / showroi commands.

        Acceptable forms:
        - setroi left=400 top=1200 width=1300 height=1000
        - setroi 400 1200 1300 1000
        - setroi                 -> set ROI to full sensor (explicit)
        - roi left=... (alias)
        - showroi
        - roi reset
        """
        cmd = parts[0].lower()
        args = parts[1:]

        # show current ROI
        if cmd in ('showroi',) or (cmd in ('roi', 'setroi') and len(args) == 1 and args[0].lower() == 'show'):
            if self.last_roi:
                l, t, w, h = self.last_roi
                print(f"[STAGE] Current ROI (last set): left={l}, top={t}, width={w}, height={h}")
            else:
                print("[STAGE] No ROI set via CLI yet.")
            return

        # If user typed just "setroi" or "roi" with no args -> set full sensor ROI explicitly
        if cmd in ('setroi', 'roi') and len(args) == 0:
            try:
                # Try no-arg call first (in case camera wrapper supports it)
                try:
                    self.camera_app.set_roi()
                    print("[STAGE] ROI set to full sensor (via camera_app.set_roi()).")
                    return
                except TypeError as e:
                    print("[STAGE] ROI wasnt set to full sensor (do it explicit).")
                    return
            except Exception as e:
                print(f"[STAGE] Failed to set ROI to full sensor: {e}")
                return

        # reset ROI (clear) via explicit keyword (does not touch camera)
        if cmd in ('roi', 'setroi') and len(args) == 1 and args[0].lower() in ('reset', 'default', 'clear'):
            self.last_roi = None
            print("[STAGE] ROI reset (no ROI will be applied by CLI).")
            return

        # If key=value form
        kv = self._parse_kv_parts(args)
        if kv:
            left = self._try_int(kv.get('left') or kv.get('l'))
            top = self._try_int(kv.get('top') or kv.get('t'))
            width = self._try_int(kv.get('width') or kv.get('w'))
            height = self._try_int(kv.get('height') or kv.get('h'))

        # If positional numbers provided: setroi 400 1200 1300 1000
        elif len(args) >= 4 and all(p.lstrip('-').isdigit() for p in args[:4]):
            left = int(args[0])
            top = int(args[1])
            width = int(args[2])
            height = int(args[3])
        else:
            print("[STAGE] Invalid ROI command. Use: setroi left=<l> top=<t> width=<w> height=<h> OR setroi <l> <t> <w> <h> OR just `setroi` to set full sensor.")
            return

        # validate numbers
        if None in (left, top, width, height):
            print("[STAGE] Invalid ROI values. Make sure left/top/width/height are integers.")
            return

        # apply ROI to camera app
        try:
            # Note: Assumes camera_app.set_roi(left, top, width, height) exists
            self.camera_app.set_roi(left, top, width, height)
            self.last_roi = (left, top, width, height)
            print(f"[STAGE] ROI set: left={left}, top={top}, width={width}, height={height}")
        except Exception as e:
            print(f"[STAGE] Failed to set ROI: {e}")


    # ---------------------- main processor ----------------------
    def process(self, command):
        try:
            cmd = command.strip()
            if not cmd:
                return

            parts = re.split(r"\s+", cmd)
            key = parts[0].lower()

            # ROI commands
            if key in ('roi', 'setroi', 'set_roi', 'showroi'):
                self._handle_roi(parts)
                return

            # show current positions
            if key == 'pos':
                x = self.stage_app.get_pos('x')
                y = self.stage_app.get_pos('y')
                z = self.stage_app.get_pos('z')
                print(f"[STAGE] Current position: X={x}nm, Y={y}nm, Z={z}nm")
                return

            # autofocus handling
            if key == 'autofocus' or key.startswith('autofocus'):
                # rebuild normalized command parts for autofocus parsing
                axis = 'x'
                scan_range = None
                step_size = None
                enable_plot = False

                for p in parts[1:]:
                    if p in ('x', 'y', 'z'):
                        axis = p
                    elif p.startswith('range='):
                        scan_range = int(p.split('=', 1)[1])
                    elif p.startswith('step='):
                        step_size = int(p.split('=', 1)[1])
                    elif p == 'noplot':
                        enable_plot = False

                self.autofocus.run_autofocus(axis, scan_range, step_size, enable_plot)
                return

            if key == 'autofocus_save':
                self.autofocus.save_results()
                return

            # absolute move: 'x 5000'
            if key in ('x', 'y', 'z') and len(parts) == 2:
                axis = key
                pos = self._try_int(parts[1])
                if pos is None:
                    print(f"[STAGE] Error: Invalid position value '{parts[1]}'")
                    return
                print(f"[STAGE] Moving {axis.upper()} to {pos}nm...")
                self.stage_app.move_abs(axis, pos)
                actual = self.stage_app.get_pos(axis)
                print(f"[STAGE] Done. {axis.upper()} is now at {actual}nm")
                return

            # move multiple axes: 'move x=5000 y=3000'
            if key == 'move':
                for p in parts[1:]:
                    if '=' in p:
                        axis, pos = p.split('=', 1)
                        axis = axis.strip().lower()
                        if axis in ('x', 'y', 'z'):
                            val = self._try_int(pos)
                            if val is None:
                                print(f"[STAGE] Error: Invalid position '{pos}'")
                            else:
                                print(f"[STAGE] Moving {axis.upper()} to {val}nm...")
                                self.stage_app.move_abs(axis, val)
                        else:
                            print(f"[STAGE] Error: Invalid axis '{axis}'")
                return

            # relative move: 'rel x=500'
            if key == 'rel':
                for p in parts[1:]:
                    if '=' in p:
                        axis, shift = p.split('=', 1)
                        axis = axis.strip().lower()
                        if axis in ('x', 'y', 'z'):
                            shift_val = self._try_int(shift)
                            if shift_val is None:
                                print(f"[STAGE] Error: Invalid shift '{shift}'")
                            else:
                                print(f"[STAGE] Moving {axis.upper()} by {shift_val}nm...")
                                self.stage_app.move_rel(axis, shift_val)
                        else:
                            print(f"[STAGE] Error: Invalid axis '{axis}'")
                return

            # help
            if key == 'help':
                self._print_help()
                return

            # quit/exit are handled by the input thread in the main app
            print(f"[STAGE] Unknown command: '{command}'. Type 'help' for usage.")

        except Exception as e:
            print(f"[STAGE] Error processing command '{command}': {e}")

    def _print_help(self):
        print('\n' + '='*70)
        print('AVAILABLE COMMANDS')
        print('='*70)
        print('pos                    - Show current X, Y, Z positions')
        print('x <pos>                - Move X axis to position (nm)')
        print('y <pos>                - Move Y axis to position (nm)')
        print('z <pos>                - Move Z axis to position (nm)')
        print('move x=<p> y=<p>       - Move multiple axes (nm)')
        print('rel x=<shift>          - Relative move (nm, can be negative)')
        print('\n--- AUTOFOCUS COMMANDS ---')
        print('autofocus              - Run autofocus on X-axis (default)')
        print('autofocus <axis>       - Run autofocus on specific axis (x/y/z)')
        print('autofocus x range=20000 step=1000  - Custom parameters')
        print('autofocus noplot       - Run without live plot')
        print('autofocus_save         - Save last autofocus results to file')
        print('\n--- ROI COMMANDS ---')
        print('setroi left=<l> top=<t> width=<w> height=<h>  - key=value form')
        print('setroi <l> <t> <w> <h>                        - positional form')
        print('setroi                                        - set ROI to full sensor (no args)')
        print('showroi                                       - show last ROI set via CLI (or none)')
        print('roi reset / roi clear / roi default           - clear last CLI ROI (does NOT change camera)')
        print('\nNotes:')
        print('  • Calling "setroi" with no arguments will explicitly set the camera ROI to the full sensor')
        print('    by querying the sensor size and applying left=0, top=0, width=<sensor_w>, height=<sensor_h>.')
        print('  • "roi reset" only clears the stored CLI ROI (so future partial setroi updates won\'t merge)\n'
            '    — it does NOT modify the camera unless you explicitly call setroi afterwards.')
        print('\nhelp                   - Show this help')
        print('quit / exit            - Stop application')
        print('='*70 + '\n')

