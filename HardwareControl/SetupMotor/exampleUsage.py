# example_usage.py
import argparse
import time
from HardwareControl.SetupMotor.xyzStageApp import XYZStageApp
from HardwareControl.SetupMotor.mockStage import MockXYZStage

# from smaract_stage import SmarActXYZStage  # uncomment when you want hardware usage

def main(use_mock=True):
    if use_mock:
        stage = MockXYZStage({'x': 1000, 'y': 2000, 'z': 0})
    else:
        from HardwareControl.SetupMotor.smartactStage import SmarActXYZStage
        stage = SmarActXYZStage()

    app = XYZStageApp(stage)

    try:
        app.move_abs('x', 0)
        app.move_abs('y', 0)

        print("start positions:", app.get_pos('x'), app.get_pos('y'), app.get_pos('z'))
        app.move_rel('x', 500)
        app.move_abs('y', 3000)
        print("after moves:", app.get_pos('x'), app.get_pos('y'), app.get_pos('z'))

        # simple grid example
        xp = [0, ]
        yp = [0, 1000, 10000, 100000, 1000000, 2000000, 10000000]
        for x, y, xr, yr in app.scan_grid('x', 'y', xp, yp):
            print(f"scanned ({x},{y}) -> readback ({xr},{yr})")
            time.sleep(1)
    finally:
        app.move_abs('x', 0)
        app.move_abs('y', 0)
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use real SmarAct stage instead of mock")
    args = parser.parse_args()
    main(use_mock=not args.real)
