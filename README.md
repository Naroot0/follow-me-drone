# Follow-Me Camera Drone

A small camera drone that follows and films one person from behind and above using only its onboard camera. Nothing is worn or carried by the person. The drone is built for hiking trails in rugged terrain where GPS (Global Positioning System) is unreliable, so the follow loop and every failsafe work without it.

This is Project 1 of a two-project program. Project 2 (a communication-free swarm simulator) is deferred until this one reaches v1.0.

## Phase 1 success criteria

Final numbers, set 2026-11-04. Phase 1 is done when every line below is true in two separate outdoor sessions.

| # | Criterion | Pass line |
|---|-----------|-----------|
| 1 | Acquisition | With the person standing 2 to 4 m in front of the drone after takeoff, the drone locks onto them within **5 s** and signals the lock (LED or tone). With a second person in frame it locks the one nearest the frame center and never switches targets except after a declared loss. |
| 2 | Station hold | Holds station **2.5 m** behind and above a walking person, **plus or minus 0.5 m**, for **10 continuous minutes** on an open trail with no pilot input after takeoff, recording video of the person the whole time. Hard floor **1.5 m**. |
| 3 | Occlusion recovery | Reacquires the same person on its own after they are hidden for up to **5 s**. Longer losses trigger hover, then land in place. |
| 4 | Obstacle stop | When an obstacle comes within **1.5 m**, the drone hovers, sidesteps, and resumes following without pilot input. |
| 5 | GPS loss | Recovers from GPS loss with no jump in position and no change in behavior, logs the source switch, and reacquires the target automatically, or lands safely for retrieval. |
| 6 | Orbit | While the person stands still, orbits a half circle around them at the set distance, keeping them framed. |
| 7 | Evidence | A 2 to 3 minute video shows following, one occlusion and recovery, and the orbit. A written report gives measured results: distance error over time, lost-track events, recovery times. |

Operating envelope for Phase 1: one person, daylight, wind under 5 m/s measured at the launch point with a handheld anemometer, open ground through hiking trails. No night, rain, snow deeper than 25 mm (1 in), crowds, or indoor flight. Snow operation is out of scope for every phase. See `docs/PLAN.md` for scope, deferred items, and the stage schedule.

## Metrics reported

Detection rate, track continuity (longest and mean unbroken track), reacquisition accuracy, false-lock rate, end-to-end latency (camera frame to velocity command), frames per second on the Jetson, distance error over time against the set station distance, and obstacle-stop response time (first obstacle reading inside the stop distance to zero commanded velocity).

## Hardware

| Part | Choice |
|------|--------|
| Airframe | GEPRC MOZ7 V2 (7-inch, 336 mm wheelbase, 750 g stock), bind-and-fly |
| Flight controller | TAKER H743 BT (STM32H743), ArduPilot target |
| Receiver | NewBeeDrone BeeCeiver ELRS 2.4 GHz diversity (swapped in for the TBS Nano) |
| Companion computer | NVIDIA Jetson Orin Nano Super developer kit, 256 GB NVMe |
| Camera | ELP AR0234 color global-shutter USB, 100 degree lens |
| Position hold without GPS | MicoAir MTF-01 optical flow plus 8 m rangefinder |
| Obstacle sensors | Four TF-Luna time-of-flight rangefinders on I2C (sides, rear, up) |
| Jetson power | 6S pack to XT60 Y splitter to 24 V to 12 V 5 A converter to barrel jack (zero solder) |
| Batteries | Three RDQ 6S 1300 mAh LiPo for stage 2; 6S 8000 mAh class Li-Ion for stage 3 onward |

Software stack: ArduPilot (Copter) on the flight controller, MAVLink (Micro Air Vehicle Link) between the Jetson and the flight controller, Python on the Jetson with a YOLO-class person detector exported to TensorRT, a ByteTrack-class tracker, and a pretrained re-identification embedding model. All flight behavior is rule-based; the models answer only "where is the person and is it the same person."

## Repository layout

```
README.md          what the project is, the pass line, how to run
docs/PLAN.md       the living project plan (scope, stages, task breakdown)
DECISIONS.md       decision log, one dated entry per decision
RISKS.md           risk register, updated at each stage gate
FLIGHT_LOG.md      one row per flight or simulation run
config/            every tunable parameter, named, in one file
src/perception/    detector, tracker, re-ID, enrollment
src/estimation/    bounding box to range and bearing
src/control/       velocity commands, station hold, orbit, state machine
src/safety/        watchdog, geofence, override check
src/logging/       per-flight CSV logs
sim/               ArduPilot SITL (software in the loop) setup and fake-target scripts
scripts/           setup and flashing helpers
logs/              flight logs (gitignored except an example)
```

## Running the simulator

Stage 1 deliverable. Instructions land here with release v0.1.

## Running on hardware

Stage 3 deliverable. Instructions land here with release v0.3.

## Releases

| Tag | Gate | Target date |
|-----|------|-------------|
| v0.1 | Software: SITL holds station on a fake target; detector at 15 fps or better on the Jetson | Nov 11 |
| v0.2 | Airframe: manual flight with radio override confirmed on ArduPilot | Nov 11 |
| v0.3 | Data and GPS-free hold: 60 min of trail footage; two-minute Loiter with GPS off | Nov 18 |
| v0.9 | Closed loop: one-minute untethered station hold on a person | Nov 25 |
| v1.0 | Flight and report: all criteria above met in two sessions | Dec 2 |

## Authorship

AI tools do lookup, summary, critique, and boilerplate. Every design decision, decision-log entry, and sentence of results is written by the author. The author must be able to explain and reproduce every section of the report.

## Privacy

Flight footage and labeled crops are stored on the author's machine, not in this repository, because trail footage includes bystanders. Only the author's own footage appears in the demo video and report.

## Regulatory

Recreational flight under 49 USC 44809. TRUST (The Recreational UAS Safety Test) certificate held by the pilot. Aircraft registered with the FAA (Federal Aviation Administration) and marked with the registration number. Remote ID approach is an open decision due Nov 11 (the MOZ7 V2 ships without Remote ID; see DECISIONS.md #19). Flights stay below 400 ft AGL (above ground level), within visual line of sight, in Class G airspace or with LAANC authorization.
