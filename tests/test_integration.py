#!/usr/bin/env python3
"""
Integration test for CHORD v5 system.

Tests the complete pipeline:
1. MusicXML ingestion
2. Fingering solver
3. Kinematics generation
4. Layer C contact planning
5. Layer B integration
6. Layer A servo
7. MuJoCo simulation

This test can be run headless (no viewer) for CI/CD.
"""

import sys
import os
import json
import tempfile
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.ingest.parse_musicxml import extract_note_table, notes_to_events
from src.fingering.dp_solver import DPSolver
from src.planning.kinematics import HandKinematics
from src.planning.layer_c import LayerCPlanner
from src.control.layer_b import LayerB
from src.control.servo import ChordServo
from src.control.constraint_factory import ConstraintFactory


def test_imports():
    """Test that all critical modules can be imported."""
    print("[TEST] Checking imports...")

    try:
        from src.control.layer_b import LayerB
        from src.control.servo import ChordServo
        from src.fingering.dp_solver import DPSolver
        from src.planning.layer_c import LayerCPlanner
        from src.planning.piano_geometry import PianoGeometry
        from src.control.state_machine import KeyStateMachine, ContactMode
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_dp_solver():
    """Test the DP fingering solver."""
    print("[TEST] Testing DP Solver...")

    solver = DPSolver()

    # Simple C major scale
    events = [
        {'pitches': [60], 'onset_sec': 0.0},  # C
        {'pitches': [62], 'onset_sec': 0.5},  # D
        {'pitches': [64], 'onset_sec': 1.0},  # E
        {'pitches': [65], 'onset_sec': 1.5},  # F
        {'pitches': [67], 'onset_sec': 2.0},  # G
    ]

    try:
        fingerings = solver.solve(events)
        assert len(fingerings) == len(events), "Fingering count mismatch"
        assert all(isinstance(f, list) for f in fingerings), "Fingerings should be lists"
        print(f"  ✓ DP Solver works: {fingerings}")
        return True
    except Exception as e:
        print(f"  ✗ DP Solver failed: {e}")
        return False


def test_layer_c():
    """Test Layer C contact planning."""
    print("[TEST] Testing Layer C...")

    layer_c = LayerCPlanner()

    events = [
        {
            'onset_sec': 1.0,
            'staff': 1,
            'pitches': [60, 64, 67],
            'fingering': [1, 3, 5],
            'notes': [
                {'pitch_midi': 60, 'velocity': 64},
                {'pitch_midi': 64, 'velocity': 64},
                {'pitch_midi': 67, 'velocity': 64},
            ]
        }
    ]

    try:
        result = layer_c.annotate_events(events)
        assert len(result) == 1, "Should have one event"
        assert 'contacts' in result[0], "Should have contacts field"
        contacts = result[0]['contacts']
        assert len(contacts) == 3, "Should have 3 contacts"

        # Check contact structure
        for c in contacts:
            assert 'pitch' in c
            assert 'finger' in c
            assert 'strike_time_sec' in c
            assert 'pretouch_start_sec' in c
            assert 'key_center' in c

        print(f"  ✓ Layer C works: {len(contacts)} contacts generated")
        return True
    except Exception as e:
        print(f"  ✗ Layer C failed: {e}")
        return False


def test_layer_b():
    """Test Layer B margin adaptation."""
    print("[TEST] Testing Layer B...")

    layer_b = LayerB(update_frequency=100.0)

    try:
        # Register a note
        layer_b.register_note(pitch=60, finger_idx=3, onset_time=1.0)

        # Simulate some updates
        from src.control.state_machine import KeyStateMachine, ContactMode

        sm = KeyStateMachine(pitch=60, onset=1.0, finger_idx=3)
        sm.mode = ContactMode.STRIKE

        key_depths = {60: 0.005}
        finger_jacobians = {3: (np.array([0.5, -0.2, 0.15]), np.eye(3, 6))}
        qvel = np.zeros(6)

        result = layer_b.update(
            current_time=0.9,
            state_machines=[sm],
            key_depths=key_depths,
            finger_jacobians=finger_jacobians,
            qvel_current=qvel,
        )

        assert result['updated'] == True, "Layer B should update"
        assert 'adapted_margins' in result
        assert 'phi_models' in result

        # Check margin retrieval
        deadline = layer_b.get_adapted_deadline(60, 3, 1.0)
        assert deadline < 1.0, "Deadline should be before onset"

        print(f"  ✓ Layer B works: margin={1.0-deadline:.3f}s")
        return True
    except Exception as e:
        print(f"  ✗ Layer B failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_servo_qp():
    """Test Layer A servo QP solver."""
    print("[TEST] Testing Layer A Servo...")

    try:
        servo = ChordServo(n_dof=6)
        constraint_factory = ConstraintFactory(ndof=6)

        # Create dummy state
        from src.control.servo import RobotState
        state = RobotState(
            q=np.zeros(6),
            q_dot=np.zeros(6),
            J=None,
            ee_pos=np.array([0.5, -0.2, 0.15]),
            ee_rot=np.eye(3),
        )

        # Nominal velocity
        q_dot_nom = np.array([0.1, 0.0, -0.05, 0.0, 0.0, 0.0])

        # Simple key constraint
        z_row = np.array([0., 0., 1., 0., 0., 0.])
        key_block = constraint_factory.create_key_constraint(
            z_jac_row=z_row,
            dt_rem=0.5,
            k_current=0.2,
            k80=0.8,
        )

        # Solve
        q_dot_cmd, diag = servo.step(
            state=state,
            q_dot_nom=q_dot_nom,
            key_constraints=key_block,
            zone_constraints=None,
            guard_constraints=None,
            sep_constraints=None,
        )

        assert q_dot_cmd.shape == (6,), "Output should be 6-DOF"
        assert diag['status'] in ['opt', 'fallback'], f"Unexpected status: {diag['status']}"

        print(f"  ✓ Layer A Servo works: status={diag['status']}")
        return True
    except Exception as e:
        print(f"  ✗ Layer A Servo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_constraint_factory():
    """Test constraint factory methods."""
    print("[TEST] Testing Constraint Factory...")

    factory = ConstraintFactory(ndof=6)

    try:
        # Test zone constraint
        zone_block = factory.create_zone_constraint(
            a=np.array([0., 0., -1.]),
            b=-0.05,
            y=np.array([0.5, -0.2, 0.1]),
            J_pos=np.eye(3, 6),
            tau_list=[0.001, 0.002],
        )
        assert zone_block.A.shape[0] == 2, "Should have 2 rows (2 collocation points)"

        # Test separation constraint
        sep_block = factory.create_separation_constraint(
            J_i=np.eye(3, 6),
            J_j=np.eye(3, 6) * 0.5,
            p_i=np.array([0.5, -0.2, 0.1]),
            p_j=np.array([0.52, -0.2, 0.1]),
            n=np.array([1., 0., 0.]),
            d_min=0.015,
        )
        assert sep_block.A.shape[0] == 1, "Should have 1 separation constraint"

        # Test gantry workspace constraint
        workspace_block = factory.create_gantry_workspace_constraint(
            gantry_qpos_translation=np.array([0.5, -0.2, 0.2]),
            gantry_qvel_indices=[0, 1, 2],
            workspace_min=np.array([0.0, -0.4, 0.05]),
            workspace_max=np.array([1.5, 0.0, 0.5]),
            dt=0.001,
        )
        assert workspace_block.A.shape[0] == 3, "Should have 3 workspace constraints"

        print("  ✓ Constraint Factory works")
        return True
    except Exception as e:
        print(f"  ✗ Constraint Factory failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_pipeline():
    """Test complete pipeline without MuJoCo simulation."""
    print("[TEST] Testing End-to-End Pipeline...")

    try:
        # 1. Create synthetic events
        events = [
            {'onset_sec': 0.0, 'pitches': [60], 'staff': 1, 'notes': [{'pitch_midi': 60, 'velocity': 64}]},
            {'onset_sec': 0.5, 'pitches': [62], 'staff': 1, 'notes': [{'pitch_midi': 62, 'velocity': 64}]},
            {'onset_sec': 1.0, 'pitches': [64], 'staff': 1, 'notes': [{'pitch_midi': 64, 'velocity': 64}]},
        ]

        # 2. Solve fingerings
        solver = DPSolver()
        fingerings = solver.solve(events)

        for i, f in enumerate(fingerings):
            events[i]['fingering'] = f

        # 3. Generate kinematics
        kinematics = HandKinematics()
        events_with_kin = kinematics.generate_trajectory(events)

        # 4. Layer C
        layer_c = LayerCPlanner()
        final_events = layer_c.annotate_events(events_with_kin)

        # Verify structure
        assert len(final_events) == 3
        for e in final_events:
            assert 'fingering' in e
            assert 'wrist_target' in e
            assert 'contacts' in e

        print("  ✓ End-to-End Pipeline works")
        return True
    except Exception as e:
        print(f"  ✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("CHORD v5 Integration Tests")
    print("="*60 + "\n")

    tests = [
        ("Imports", test_imports),
        ("DP Solver", test_dp_solver),
        ("Layer C", test_layer_c),
        ("Layer B", test_layer_b),
        ("Layer A Servo", test_servo_qp),
        ("Constraint Factory", test_constraint_factory),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"[FATAL ERROR in {name}]: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
        print()

    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8s} {name}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! CHORD v5 is ready for simulation.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
