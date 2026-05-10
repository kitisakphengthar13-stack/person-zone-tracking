# Pose-Based Sit-Down Tracking

A portfolio-oriented computer vision prototype for work-zone posture compliance monitoring. This branch extends the original person-zone tracking project with YOLO pose tracking, keypoint-based posture classification, duration-based sitting violation confirmation, visualization overlays, and optional JSONL posture event logging.

The intended base pose model is `yolo26n-pose`, configured by default as:

```text
models/yolo26n-pose.pt
```

This repository does not include model weights, videos, generated outputs, logs, or Git metadata. It does not download YOLO weights automatically.

> This is a prototype, not a certified safety system. It is not a medical or ergonomic assessment tool. Current validation is automated code-level validation only; no real webcam, video, or YOLO model validation has been performed, and no real YOLO weights were loaded in tests.

---

## Project Purpose

The goal of this branch is to detect tracked people who appear to be sitting in a configured work zone where standing is expected.

The system uses YOLO pose keypoints as the main approach. It does not rely on simple object-detection classes such as `sitting` or `standing`.

A sitting violation is confirmed only when the same tracked person remains classified as sitting for more than 30 seconds.

Typical prototype use cases:

- Work-zone posture compliance monitoring
- Duration-based sit-down behavior tracking
- Zone-aware worker behavior experiments
- Portfolio demonstration of YOLO pose, tracking, polygon zones, temporal debouncing, and event logging

---

## Runtime Pipeline

When `posture.enabled` is `false`, the original person-zone tracking pipeline remains available:

```text
frame
-> YOLO object detection/tracking
-> polygon zone matching
-> dwell-time tracking
-> visualization and optional output video
```

When `posture.enabled` is `true`, the posture pipeline is used:

```text
frame
-> YOLO pose tracking
-> PosePerson extraction
-> keypoint quality validation
-> posture classification: standing / sitting / unknown
-> polygon zone matching
-> sitting duration tracking
-> 30-second violation confirmation
-> visualization overlay
-> optional JSONL event logging
```

Zone matching is reused from the original project. `PosePerson` exposes `bbox`, `class_name`, `confidence`, and `track_id`, so existing polygon zone logic can match pose-tracked people without needing a separate zone system.

---

## Key Modules

| File | Purpose |
|---|---|
| `src/pose_types.py` | Pure dataclasses and enums for `Keypoint`, `PosePerson`, `PostureState`, and `PostureStatus` |
| `src/pose_detector.py` | YOLO pose adapter and pure conversion from Ultralytics-like pose results to `PosePerson` objects |
| `src/posture_classifier.py` | Frame-level keypoint quality checks and posture classification into `standing`, `sitting`, or `unknown` |
| `src/posture_events.py` | Dataclasses/constants for posture violation state and emitted events |
| `src/posture_violation_engine.py` | Duration/debounce engine that confirms sitting violations after more than 30 seconds |
| `src/posture_event_logger.py` | Optional JSONL serialization and file append helper for emitted posture events |
| `src/visualizer.py` | Existing OpenCV overlay with optional posture labels |
| `src/main.py` | Runtime branching between original detection mode and posture mode behind `posture.enabled` |

Original project modules still used:

| File | Purpose |
|---|---|
| `src/config.py` | YAML/CLI config loading and validation |
| `src/zone_manager.py` | Polygon zone loading and bbox-to-zone matching |
| `src/dwell_time.py` | Per-zone dwell-time accumulation |
| `src/video_source.py` | Webcam/video frame source and timestamps |
| `src/tracker.py` | Thin wrapper around Ultralytics `model.track(..., persist=True)` |

---

## Configuration

Edit `configs/app.yaml` to configure the original tracker and the posture branch.

Posture mode is disabled by default:

```yaml
posture:
  enabled: false
  model_path: models/yolo26n-pose.pt
  model_type: yolo_pose
  person_class_name: person
  required_state: standing
  violation_state: sitting
  min_violation_seconds: 30.0
  clear_after_seconds: 2.0
  unknown_grace_seconds: 3.0
  max_missing_track_seconds: 2.0
  keypoints:
    min_confidence: 0.35
    min_required_points: 6
  zones:
    ignore_outside_zones: true
    posture_required_zone_ids: []
  events:
    enabled: true
    log_events: false
    event_log_path: data/outputs/posture_events.jsonl
```

Important fields:

| Field | Meaning |
|---|---|
| `posture.enabled` | Enables the pose-based posture pipeline when `true` |
| `posture.model_path` | Intended YOLO pose model path, default `models/yolo26n-pose.pt` |
| `posture.min_violation_seconds` | Sitting must persist for more than this duration before an active violation event is emitted |
| `posture.clear_after_seconds` | Standing/non-violation must persist this long before clearing an active violation |
| `posture.unknown_grace_seconds` | Short unknown/occluded intervals do not immediately reset sitting state |
| `posture.max_missing_track_seconds` | Short missing-track gaps do not immediately clear state |
| `posture.events.enabled` | Enables the posture event subsystem |
| `posture.events.log_events` | Writes emitted posture events to JSONL when `true` |
| `posture.events.event_log_path` | JSONL output path for posture events |

To enable posture mode:

```yaml
posture:
  enabled: true
  model_path: models/yolo26n-pose.pt
  min_violation_seconds: 30.0
```

Then run:

```bash
python src/main.py --config configs/app.yaml
```

The model file must exist locally. This repository does not download model weights.

---

## Posture States

The classifier returns one frame-level posture state for each `PosePerson`:

| State | Meaning |
|---|---|
| `standing` | Keypoints suggest an upright standing posture |
| `sitting` | Keypoints suggest seated hip/knee geometry |
| `unknown` | Keypoints are missing, low-confidence, occluded, or geometrically ambiguous |

The implementation intentionally uses `unknown` when evidence is insufficient. Lower-body occlusion, low-confidence hips/knees, upper-body-only detections, or ambiguous geometry should not be forced into sitting.

---

## Visualization Labels

The visualization overlay appends posture text to tracked person labels when posture data is available.

Example labels:

```text
ID 2 | standing
ID 4 | sitting
ID 4 | sitting 12.4s
ID 4 | sitting violation 31.2s
ID 7 | unknown posture
```

The first implementation keeps the overlay compact. It does not add a dashboard or UI panel.

---

## JSONL Posture Event Logging

Posture event logging is optional and disabled by default:

```yaml
posture:
  events:
    enabled: true
    log_events: false
    event_log_path: data/outputs/posture_events.jsonl
```

When enabled, the logger writes only emitted posture violation events. It does not write one row per frame.

Example JSONL event row:

```json
{"duration_seconds":31.2,"emitted_at":41.2,"ended_at":null,"event_type":"sitting_violation_active","posture_state":"sitting","started_at":10.0,"state":"active","track_id":7,"zone_id":"zone_1","zone_name":"Work Zone"}
```

Typical event types:

```text
sitting_violation_active
sitting_violation_cleared
```

---

## Zone Tracking

Zones are stored in JSON format:

```json
{
  "version": 1,
  "zones": [
    {
      "id": "zone_1",
      "name": "Zone 1",
      "points": [
        [447, 119],
        [836, 118],
        [843, 439],
        [437, 439]
      ],
      "target_classes": ["person"]
    }
  ]
}
```

Zone membership uses the existing bbox center and bbox/polygon overlap logic. In posture mode, `PosePerson` bounding boxes are passed into the same zone matcher.

Sitting outside matched zones is ignored by default through:

```yaml
posture:
  zones:
    ignore_outside_zones: true
```

---

## Installation

Create and activate a Python virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Place local YOLO weights in `models/`.

Example:

```text
models/yolo26n-pose.pt
```

---

## Usage

Run all commands from the project root.

Run with YAML config:

```bash
python src/main.py --config configs/app.yaml
```

Run with video input:

```bash
python src/main.py --source-type video --source-path "D:/videos/test.mp4" --config configs/app.yaml
```

Draw zones:

```bash
python src/main.py --source-type video --source-path "D:/videos/test.mp4" --draw-zones true --zones-path assets/zones.json
```

Save output video:

```bash
python src/main.py --config configs/app.yaml --save-output true --output-path data/outputs/result.mp4
```

---

## Automated Tests

Run the automated code-level test suite:

```bash
pytest tests -p no:cacheprovider --basetemp=tests_tmp_posture_phase7
```

Current result:

```text
67 passed
```

The tests cover config parsing, pose dataclasses, pose result conversion with fake Ultralytics-like objects, synthetic-keypoint posture classification, duration-based violation confirmation, visualization label helpers, JSONL serialization, and runtime branch helpers.

The tests do not load real YOLO weights and do not validate real webcam or video behavior.

---

## Limitations

- Validation is automated code-level validation only.
- No real webcam, video, or model-weight validation has been performed yet.
- No real YOLO weights were loaded in tests.
- No detection accuracy claims are made.
- This is not a certified safety system.
- This is not a medical or ergonomic assessment tool.
- 2D pose-based sitting detection is uncertain under occlusion.
- Camera angle, lighting, tracker ID switches, low FPS, and missing keypoints can affect behavior.
- Crouching, bending, squatting, leaning, and sitting can be visually ambiguous.
- Lower-body occlusion can force `unknown` posture rather than a confident state.
- A 30-second violation depends on stable tracking IDs; ID switches can interrupt confirmation.

---

## Future Runtime Validation

The next step is runtime validation with:

- actual `yolo26n-pose` weights
- a recorded sample video or webcam feed
- observed FPS and hardware details
- qualitative false positive and false negative review
- documented camera angle, lighting, zone placement, occlusion cases, and limitations

Any future README claims about accuracy or real-world performance should be based on measured evaluation, not assumptions.

---

## Notes

This branch is intended to demonstrate a practical architecture for posture-compliance experimentation:

```text
YOLO pose + tracking + polygon zones + duration-based violation confirmation
```

The original person-zone tracking behavior remains available when `posture.enabled` is `false`.
