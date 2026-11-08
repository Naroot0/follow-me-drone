# Flight log

One row per flight or simulation run. Build tag is the git tag or short commit hash. Log file links point to `logs/` (gitignored) or to the ArduPilot .bin dataflash log kept on the author's machine.

| Date | Build | Type | Site | Wind (m/s) | Pack | What was tested | Result | Log |
|------|-------|------|------|------------|------|-----------------|--------|-----|
| | | sim / bench / tethered / free | | anemometer at launch | | | | |
| 2026-11-06 | ardupilot fix-fwversion | sim | SITL default (Canberra) | n/a | n/a | SITL and QGC install; arm, Guided takeoff to 10 m, Loiter hold, Land | Pass. First Loiter attempt crashed because the simulated throttle was at minimum; passed after `rc 3 1500`. | none |
| 2026-11-07 | detector.py (uncommitted) | bench | desk, Mac Studio, Logitech webcam | n/a | n/a | YOLO11n person detector on MPS, informal walk-around, 72 s | 19.8 fps end to end; inference median 12.8 ms, p95 49.5 ms. Capture and display, not the model, set the frame rate. | none |
| 2026-11-07 | tracker.py (uncommitted) | bench | desk, Mac Studio, Logitech webcam | n/a | n/a | ByteTrack, one person, 115 s: still, lateral walks, back turned 5 s, partial occlusion behind furniture | Pass. 1 ID for all 1869 frames, 16.3 fps. Tracker settings: detector conf 0.25, new_track_thresh 0.5, track_buffer 60. An earlier run with detector conf 0.1 produced 12 IDs from fragment boxes and was rejected. | none |

| 2026-11-07 | tracker.py (uncommitted) | bench | desk, Mac Studio, Logitech webcam | n/a | n/a | ByteTrack, two people, four full crossings in 28 s | 7 IDs assigned, about one switch per crossing. Expected for a motion-only tracker; this is the re-ID hand-off case (decision 2). Baseline for the re-ID task. | none |
| 2026-11-08 | reid.py (uncommitted) | bench | desk, Mac Studio, Logitech webcam | n/a | n/a | OSNet x0.25 re-ID, one person, 114 s: poses, turns, partial occlusion, leave and re-enter | Embed 8.6 ms median per frame on MPS; 29.6 fps end to end with detector and tracker. Self-similarity (max over a 30-view gallery): min 0.71, p5 0.82, median 0.92. First attempt with a single-pose gallery and mean matching gave median 0.71 and was rejected. | none |

| 2026-11-08 | reid.py (uncommitted) | bench | desk, Mac Studio, Logitech webcam | n/a | n/a | OSNet x1.0 re-ID, same protocol, 74 s | Embed 9.0 ms median on MPS (GPU not the limit at 1 to 2 crops; Jetson will differ). Self-similarity min 0.71, p5 0.85, median 0.96. Separation from other people not yet measured; needs a second person. | none |
