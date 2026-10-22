# DOBOT ATOM MAX — ACT Pick-and-Place Progress
**Date:** 2026-10-21  
**System:** ATOM MAX humanoid, stereo head camera, ACT imitation-learning policy  
**Dataset:** 5 teleoperated HDF5 demonstrations (e.g., `episode_0.hdf5`, 1,239 frames)  
**Checkpoint:** `ModelTrain/ckpt/bottle_pick_place/policy_best.ckpt`

## 1. What worked today
- **Previously:** The autonomous inference script repeatedly reported a disabled/alarmed right-arm J6 and stopped.
- **After velocity-timing correction:** Inference kept running near the requested **50 ms / 20 Hz** cycle without the previous J6 alarm in that run. The arm made only tiny adjustments; fingers did not grasp.
- **After replacing synthetic QPOS with measured QPOS:** The robot began meaningful task execution: it **grasped a cup handed to it by a person** and **placed it within the black rectangle on the table**.
- **Not yet achieved:** Reliably autonomously approaching, picking up, and grasping the cup from its starting position on the table.

## 2. The two changes and why they matter

### Fix A — incorrect velocity denominator
**File:** `script/o4_run_inference_poc.py` (near original line 254).

Before:
```python
arm_velocity = (target_arm - arm_state) / (time.time() - cycle_start)
```
After:
```python
arm_velocity = (target_arm - arm_state) / (config.cycle_time / 1000.0)
```
**Explanation:** The original denominator was the *computation time so far in the current cycle*, not the intended time between control commands. This could generate disproportionately large speed requests (subsequently clipped). The new denominator uses the nominal 50 ms period. **The disappearance of the J6 alarm is correlated with this change, but the servo's actual fault cause is not yet proven.** Timing and motion limits still need proper validation.

### Fix B — use measured joint state for ACT observation
**File:** `script/o4_run_inference_poc.py` (near original line 226).

Before:
```python
observation["qpos"] = last_action.copy()
```
After:
```python
current_arm, current_hand, current_head = robot.get_joint_state()
observation["qpos"] = np.concatenate([
    current_arm[0:7], current_hand[0:6],
    current_arm[7:14], current_hand[6:12],
    current_head
])
```
**Explanation:** During training, `observations/qpos` contains the robot's *measured* joint configuration, while `action` contains the desired position command. During inference, the old script fed the previous **action** back as **qpos**. This is an input-distribution mismatch, particularly for the fingers: in episode 0, right-hand QPOS at frame 0 was approximately `[-165, -173, -158, -163, 2, 157]°`, while the right-hand ACTION was `[175, 174, 172, 176, 53, 146]°`. Actual QPOS feedback restores the kind of inputs the model saw during training. **The improvement after this edit strongly supports that mismatch as a major blocker, though multiple causes can coexist.**

**Important data-quality finding:** In episode 0, recorded right-finger QPOS remained numerically identical at sampled frames 0, 100, and 300 even while finger ACTION changed significantly. Check finger-state telemetry / recording separately; do not assume its absolute values are calibrated.

## 3. What this does — and does not — prove
**Demonstrated:** Teleoperation → HDF5 data conversion → ACT training/checkpoint loading → live camera inference → model-to-robot control → assisted grasp → autonomous placement.

**Still unproven:** A complete **autonomous table-to-grasp-to-place** cycle, consistent grasp reliability, broader object/position generalization, and definitive resolution of the historical J6 alarm.

## 4. Engineering next steps: measure, then improve

**Step 1 — Establish a baseline (today, 10–20 min):** Freeze the current working checkpoint and script. Use the *same object, start pose, lighting, and target location* as the demonstrations. Run **10 supervised table-pick attempts** and record each attempt as: `approach reached? / fingers closed? / lift succeeded? / place succeeded?`. Count assisted handover separately. Do not modify multiple variables at once.

**Step 2 — Improve demonstrations (next data collection):** Record **~20–30 additional high-quality complete demonstrations** that **start with the cup on the table**, include clear approach → align → close fingers → lift → transport → release, and start from a repeatable pose. First keep the scene fixed; then deliberately vary the cup position and orientation within a small, documented range. If the training object was a bottle but deployment uses a different cup, collect demonstrations with the **actual deployment cup**.

**Step 3 — Retrain and compare:** Preserve today's checkpoint as a baseline. Train a new checkpoint with the expanded dataset, using unchanged camera order, joint conventions, and preprocessing. Compare **fixed-scene table-pick success out of 10** against today's baseline. Increase data and variation progressively only when the prior phase is reliable.

**Step 4 — Check controller tracking:** During supervised testing, log `right_arm_target`, `right_arm_measured`, `right_finger_target`, `right_finger_measured`, and any J6 alarm code, especially during the first 10 seconds. The previously observed right J6 target (~−6.5°) versus actual (~−3.1°) and constant finger feedback merit investigation.

## 5. Remaining control risks
- The script performs a first-action `dynamic_approach` before its normal `check_pose_protection` path. Review/guard that behavior before treating inference as production-ready.
- The 50 ms timing correction alone is not full velocity/acceleration/position limiting.
- Use supervised tests with your established E-stop and fault-response procedure; never repeatedly reset or retry an alarm without diagnosing it.

## One-sentence conclusion
**We validated the ACT deployment pipeline and solved a major observation mismatch; the remaining challenge is reliable autonomous object approach and grasp, best addressed through measured fixed-scene tests and targeted additional complete demonstrations.**
