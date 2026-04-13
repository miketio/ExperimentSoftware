# hardware_control/setup_motor/test_hcu_stage.py
"""
Test suite for HCU3CStage (direct SCU3DControl.dll ctypes wrapper).

Run from project root:
    python -m hardware_control.setup_motor.test_hcu_stage
"""

import sys
import traceback

PASS, FAIL = "✅  PASS", "❌  FAIL"
results = {}


def banner(title):
    print("\n" + "─" * 58)
    print(f"  {title}")
    print("─" * 58)


# ─────────────────────────────────────────────────────────────
#  TEST 1 — device discovery
# ─────────────────────────────────────────────────────────────
banner("TEST 1 — SCU device discovery")
try:
    from hardware_control.setup_motor.hcu3c_stage import list_hcu_devices
    devs = list_hcu_devices()

    if devs:
        print(f"✅  {len(devs)} device(s):")
        for d in devs:
            print(f"     idx={d['device_idx']}  id={d['device_id']}"
                  f"  fw={d['firmware']}  dll={d['dll']}")
        results[1] = PASS
    else:
        print("⚠️  No SCU devices found — is the HCU-3C plugged in?")
        results[1] = FAIL
except Exception as e:
    print(f"❌  EXCEPTION in 1 — Discovery: {e}")
    traceback.print_exc()
    results[1] = FAIL


# ─────────────────────────────────────────────────────────────
#  TEST 2 — open device and read all axis positions
# ─────────────────────────────────────────────────────────────
banner("TEST 2 — open and read all axis positions")
stage = None
try:
    from hardware_control.setup_motor.hcu3c_stage import HCU3CStage

    if results.get(1) != PASS:
        raise RuntimeError("Skipped — no devices found in Test 1.")

    stage = HCU3CStage(device_idx=0)

    for ax in ('x', 'y', 'z'):
        pos    = stage.get_pos(ax)
        status = stage.get_status(ax)
        print(f"  {ax.upper()}: {pos:>12,} nm  ({pos/1000:.3f} µm)  status={status}")

    stage.print_status()
    results[2] = PASS

except Exception as e:
    print(f"❌  EXCEPTION in 2 — Open & read: {e}")
    traceback.print_exc()
    results[2] = FAIL
    if stage:
        try:
            stage.close()
        except Exception:
            pass
        stage = None


# ─────────────────────────────────────────────────────────────
#  TEST 3 — relative moves  (±10 µm per axis)
# ─────────────────────────────────────────────────────────────
banner("TEST 3 — relative moves  (±10 µm per axis)")
try:
    if stage is None:
        from hardware_control.setup_motor.hcu3c_stage import HCU3CStage
        stage = HCU3CStage(device_idx=0)
    stage.move_abs('x', 0.0)
    stage.move_abs('y', 0.0)
    stage.move_abs('z', 0.0)

    for ax in ('x', 'y', 'z'):
        before    = stage.get_pos(ax)
        stage.move_rel(ax, +10_000)
        after_fwd = stage.get_pos(ax)
        # stage.move_rel(ax, -10_000)
        after_rev = stage.get_pos(ax)
        print(f"  {ax.upper()}: {before:>10,} → {after_fwd:>10,} → {after_rev:>10,} nm")

    results[3] = PASS

except Exception as e:
    print(f"❌  EXCEPTION in 3 — Relative moves: {e}")
    traceback.print_exc()
    results[3] = FAIL


# # ─────────────────────────────────────────────────────────────
# #  TEST 4 — absolute moves  (X axis, ±50 µm)
# # ─────────────────────────────────────────────────────────────
# banner("TEST 4 — absolute moves  (X axis, ±50 µm)")
# try:
#     if stage is None:
#         from hardware_control.setup_motor.hcu3c_stage import HCU3CStage
#         stage = HCU3CStage(device_idx=0)

#     stage.move_abs('x', 50_000)
#     p1 = stage.get_pos('x')
#     print(f"  After move_abs(+50 µm):  {p1:>12,} nm")

#     stage.move_abs('x', -50_000)
#     p2 = stage.get_pos('x')
#     print(f"  After move_abs(-50 µm):  {p2:>12,} nm")

#     stage.move_abs('x', 0)
#     p3 = stage.get_pos('x')
#     print(f"  After move_abs(0 µm):    {p3:>12,} nm")

#     results[4] = PASS

# except Exception as e:
#     print(f"❌  EXCEPTION in 4 — Absolute moves: {e}")
#     traceback.print_exc()
#     results[4] = FAIL


# # ─────────────────────────────────────────────────────────────
# #  TEST 5 — StageAdapterUM (µm interface)
# # ─────────────────────────────────────────────────────────────
# banner("TEST 5 — StageAdapterUM (µm interface)")
# try:
#     from hardware_control.setup_motor.stage_adapter import StageAdapterUM

#     if stage is None:
#         from hardware_control.setup_motor.hcu3c_stage import HCU3CStage
#         stage = HCU3CStage(device_idx=0)

#     adapter = StageAdapterUM(stage)

#     adapter.move_abs('x', 20.0)
#     px = adapter.get_pos('x')
#     print(f"  After move_abs X 20 µm:  {px:.3f} µm")

#     adapter.move_rel('y', 5.0)
#     py = adapter.get_pos('y')
#     print(f"  After move_rel Y +5 µm:  {py:.3f} µm")

#     print(f"  All positions (µm): {adapter.get_pos_all()}")

#     # adapter.move_abs('x', 0.0)
#     # adapter.move_abs('y', 0.0)

#     results[5] = PASS

except Exception as e:
    print(f"❌  EXCEPTION in 5 — Adapter (µm): {e}")
    traceback.print_exc()
    results[5] = FAIL
finally:
    if stage is not None:
        try:
            stage.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 58)
print("  RESULTS")
print("─" * 58)
test_names = {
    1: "Discovery",
    2: "Open & read",
    3: "Relative moves",
    4: "Absolute moves",
    5: "Adapter (µm)",
}
passed = sum(1 for v in results.values() if v == PASS)
for n, name in test_names.items():
    print(f"  {results.get(n, '⚠️   NOT RUN')}  {n} — {name}")
print("─" * 58)
print(f"  {passed} / {len(test_names)} passed")

sys.exit(0 if passed == len(test_names) else 1)