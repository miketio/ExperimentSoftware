#!/usr/bin/env python3
"""
test_hcu_stage.py
─────────────────
Smoke-test for SmarAct HCU-3C via pylablib SCU interface.
Run with ONLY the HCU connected.

All moves are small (≤ 50 µm) so the test is safe to run
without worrying about hitting physical limits.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pylablib.devices import SmarAct


SEP = "─" * 58

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ─────────────────────────────────────────────────────────────
# Test 1 — device discovery
# ─────────────────────────────────────────────────────────────

def test_discover():
    section("TEST 1 — SCU device discovery")

    devices = SmarAct.list_scu_devices()
    if not devices:
        print("❌  No SCU devices found")
        return False

    print(f"✅  {len(devices)} SCU device(s):")
    for d in devices:
        print(f"     device_id={d.device_id}  "
              f"fw={d.firmware_version}  dll={d.dll_version}")
    return True


# ─────────────────────────────────────────────────────────────
# Test 2 — open, read positions, close
# ─────────────────────────────────────────────────────────────

def test_open_and_read():
    section("TEST 2 — open and read all axis positions")

    from hardware_control.setup_motor.hcu_stage import HCU3CStage

    stage = HCU3CStage(verbose=True)
    try:
        stage.print_status()
        for ax in ("x", "y", "z"):
            pos = stage.get_pos(ax)
            print(f"  {ax.upper()}: {pos:>14,} nm   ({pos/1e6:>9.3f} mm)")
        print("✅  Open and readback OK")
        return True
    except Exception as e:
        print(f"❌  {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        stage.close()


# ─────────────────────────────────────────────────────────────
# Test 3 — relative moves ±10 µm on every axis
# ─────────────────────────────────────────────────────────────

def test_relative_moves():
    section("TEST 3 — relative moves  (±10 µm per axis)")

    from hardware_control.setup_motor.hcu_stage import HCU3CStage

    stage   = HCU3CStage(verbose=False)
    STEP_NM = 10_000    # 10 µm
    TOL_NM  = 5_000     # 5 µm tolerance
    all_ok  = True

    try:
        for ax in ("x", "y", "z"):
            before = stage.get_pos(ax)
            print(f"\n  {ax.upper()} start: {before:,} nm  ({before/1e6:.3f} mm)")

            # +10 µm
            stage.move_rel(ax, +STEP_NM)
            after = stage.get_pos(ax)
            err   = abs(after - (before + STEP_NM))
            ok    = err < TOL_NM
            print(f"  +10 µm → {after:,} nm   err={err/1e3:.2f} µm  {'✅' if ok else '❌'}")
            if not ok: all_ok = False

            # −10 µm (back)
            stage.move_rel(ax, -STEP_NM)
            back  = stage.get_pos(ax)
            err   = abs(back - before)
            ok    = err < TOL_NM
            print(f"  −10 µm → {back:,} nm   err={err/1e3:.2f} µm  {'✅' if ok else '❌'}")
            if not ok: all_ok = False

    except Exception as e:
        print(f"\n❌  {e}")
        import traceback; traceback.print_exc()
        all_ok = False
    finally:
        stage.close()

    return all_ok


# ─────────────────────────────────────────────────────────────
# Test 4 — absolute moves on X axis  (±50 µm from current)
# ─────────────────────────────────────────────────────────────

def test_absolute_moves():
    section("TEST 4 — absolute moves  (X axis, ±50 µm)")

    from hardware_control.setup_motor.hcu_stage import HCU3CStage

    stage  = HCU3CStage(verbose=False)
    DELTA  = 50_000     # 50 µm
    TOL_NM = 5_000
    all_ok = True

    try:
        origin = stage.get_pos("x")
        print(f"  X origin: {origin:,} nm  ({origin/1e6:.3f} mm)")

        for label, target in [
            ("+50 µm", origin + DELTA),
            ("−50 µm", origin - DELTA),
            ("origin", origin),
        ]:
            stage.move_abs("x", target)
            actual = stage.get_pos("x")
            err    = abs(actual - target)
            ok     = err < TOL_NM
            print(f"  → {label:<8}  target={target:,}  "
                  f"actual={actual:,}  err={err/1e3:.2f} µm  "
                  f"{'✅' if ok else '❌'}")
            if not ok: all_ok = False

    except Exception as e:
        print(f"\n❌  {e}")
        import traceback; traceback.print_exc()
        all_ok = False
    finally:
        stage.close()

    return all_ok


# ─────────────────────────────────────────────────────────────
# Test 5 — StageAdapterUM  (µm interface used by the rest of app)
# ─────────────────────────────────────────────────────────────

def test_adapter():
    section("TEST 5 — StageAdapterUM (µm interface)")

    from hardware_control.setup_motor.hcu_stage   import HCU3CStage
    from hardware_control.setup_motor.stage_adapter import StageAdapterUM

    stage_nm = HCU3CStage(verbose=False)
    stage_um = StageAdapterUM(stage_nm)
    all_ok   = True

    try:
        print("  Positions via adapter:")
        for ax in ("x", "y", "z"):
            pos_um = stage_um.get_pos(ax)
            print(f"    {ax.upper()}: {pos_um:.3f} µm")

        # 5 µm round-trip on X
        before = stage_um.get_pos("x")
        stage_um.move_rel("x", 5.0)
        after  = stage_um.get_pos("x")
        stage_um.move_rel("x", -5.0)

        err = abs(after - before - 5.0)
        ok  = err < 0.01        # 10 nm at the µm scale
        print(f"\n  +5 µm move: {before:.3f} → {after:.3f} µm  "
              f"err={err*1e3:.2f} nm  {'✅' if ok else '❌'}")
        if not ok: all_ok = False

    except Exception as e:
        print(f"\n❌  {e}")
        import traceback; traceback.print_exc()
        all_ok = False
    finally:
        stage_nm.close()

    return all_ok


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═'*58}")
    print("  HCU-3C Test Suite  (pylablib SCU)")
    print(f"{'═'*58}\n")

    tests = [
        ("1 — Discovery",       test_discover),
        ("2 — Open & read",     test_open_and_read),
        ("3 — Relative moves",  test_relative_moves),
        ("4 — Absolute moves",  test_absolute_moves),
        ("5 — Adapter (µm)",    test_adapter),
    ]

    passed, failed = [], []

    for name, fn in tests:
        try:
            ok = fn()
        except Exception as e:
            print(f"❌  EXCEPTION in {name}: {e}")
            ok = False
        (passed if ok else failed).append(name)

    print(f"\n{SEP}\n  RESULTS\n{SEP}")
    for t in passed: print(f"  ✅  PASS  {t}")
    for t in failed: print(f"  ❌  FAIL  {t}")
    print(f"{SEP}")
    print(f"  {len(passed)} / {len(tests)} passed\n")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())