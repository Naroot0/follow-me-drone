> This file is the project plan of record. Risks live in `RISKS.md`, decisions in `DECISIONS.md`. Last structural revision 2026-11-04.

# Follow-Me Camera Drone — Project Plan

Oct 26, 2026 · @Pete

## Purpose and success criteria

Build a small camera drone that follows and films one person from 2.5 m behind and above using only its onboard camera with nothing worn by the person.  The drone will be used to follow the person while hiking in rugged terrain with potential GPS loss.  Later phases will add faster movement such as trail running or mountain biking, and dynamic video angles beyond directly behind the person. Snow operation is out of scope for every phase (decision 20).  

The project's phase 1 is done when all of the following are true:

- [ ] Acquisition: with the person standing 2 to 4 m in front of the drone after takeoff, the drone locks onto them within 5 seconds and signals the lock (an LED or a tone). With a second person in frame, it locks the one nearest the center and never switches targets except after a declared loss. Acquisition is the one-time lock at the start; reacquisition after occlusion is the separate criterion below.

* [ ] The drone holds station 2.5 m behind and above a walking person (plus or minus 0.5 m) for 10 continuous minutes on an open trail, with no pilot input after takeoff, and records video of the person the whole time.
* [ ] The drone recovers tracking of the same person on its own after they are hidden for up to 5 seconds (behind a tree, a corner, another person). Longer losses trigger a safe hover, then land in place. When an obstacle comes within 1.5 m, the drone hovers, adjusts, and resumes following without pilot input.
* [ ] The drone gracefully recovers from GPS loss and reacquires target automatically, or sets down safely for retrieval.
* [ ] While the person stands still, the drone orbits a half circle around them at the set distance, keeping them framed. A 2 to 3 minute video shows following, one occlusion and recovery, and the orbit.
* [ ] A GitHub repository holds the code, the project board, the risk register, the decision log, and a written report with measured results (distance error over time, lost-track events, recovery times).

Final numbers, set 2026-11-04 (decision #18): acquisition 5 s; station 2.5 m, plus or minus 0.5 m, hard floor 1.5 m; 10 continuous minutes; occlusion recovery up to 5 s; obstacle stop 1.5 m. The pass/fail line was fixed before build work started.

## Scope

**Models and AI behavior.** The AI layer has its own scope line, separate from the hardware table below.

| Area | In scope | Out of scope | Deferred to Phase 2 or later |
| --- | --- | --- | --- |
| Detector | One pretrained person detector of the YOLO class, exported to TensorRT at INT8 or FP16 precision, running at 15 frames per second or better on the Jetson with the camera at the resolution used in flight. It outputs a bounding box and a confidence for every person in frame. | No training of a detector from scratch. No detection of anything other than people: dogs, bikes, vehicles, and other objects are not detected or avoided by the detector (obstacles are handled by the time-of-flight sensors). | Fine-tuning the detector on the project's own trail footage, but only if the pretrained one measurably fails on trail backgrounds (detection rate below the threshold set in Phase 0). |
| Tracker | A standard multi-object tracker of the ByteTrack or DeepSORT class. It takes the detector's boxes and assigns each person a stable ID from frame to frame, so the follow loop can tell the enrolled person from others without re-running re-ID every frame. | No trained or end-to-end tracker; the tracker is an off-the-shelf algorithm with tunable parameters, not a model that is trained. | Re-tuning the tracker's parameters (association thresholds, track lifetime) for faster subjects. |
| Re-identification (re-ID) | One pretrained re-identification (re-ID) embedding model that turns a person crop into a vector describing their appearance. At takeoff the drone enrolls the person nearest the center of the frame by storing their vector. After a track loss it compares every visible person's vector to the stored one by cosine similarity and reacquires the best match above a threshold that is set by the author, recorded in the config file, and logged on every match. Fine-tuning this model on the project's own labeled crops starts at the Bench stage (stage 3 of the Phase 1 roadmap), once real footage exists. | No face recognition. No identity that persists across flights: the drone re-enrolls the person at every takeoff and forgets them at landing. | A gesture-based enrollment classifier, so a raised arm selects the person instead of the center-of-frame rule. Persistent identity across days, so the same person is recognized at the next takeoff without re-enrolling. |
| Behavior | All flight behavior is rule-based. The rules are: enroll the person at takeoff; follow at the set distance and angle; orbit while the person stands still; on an obstacle inside the stop distance, hover, sidestep, and resume; on track loss, hover; on prolonged loss, land. Every rule's numbers (distances, angles, timeouts, gains) live in one config file, and every rule can be stated as a single if-then sentence. Additional rules may be added as needed. | No learned controller of any kind. That means: no neural network or reinforcement-learning policy that maps sensor input to velocity commands; no end-to-end model that outputs flight commands from raw frames; no learned obstacle avoidance or path planning, including monocular depth models driving the controller; no behavior whose trigger and action cannot be written as one if-then sentence in the config file; and no control gains that adapt or self-tune during flight. The models' job ends at "where is the person and is it the same person." A hand-written rule decides what to do about it. Rules can be tested step by step in simulation, stated in plain language, and debugged from the flight log. | Front-follow while walking, climb mode, and parameters for faster subjects. Also two learned components that leave the rules in charge of the commands: (a) a trajectory predictor, a small model that forecasts the person's position two seconds ahead from their recent path and feeds that forecast to the same rule-based controller, giving anticipatory following with explainable commands; (b) for cinematic framing in a later phase, a shot-quality scorer that chooses among candidate positions the rules generate. |
| Training loop | The training loop runs in six steps. Collect footage from bench and flight sessions. Label the crops, with the tracker doing the first pass and the author correcting its mistakes. Fine-tune the re-ID model on the labeled set. Evaluate it on held-out footage. Export the result to the Jetson. Measure frame rate and accuracy on the bench. The report records the before-and-after numbers for every run. | No cloud training pipelines; fine-tuning runs on the author's computer or a single rented GPU session. No paid automated labeling services; labels come from the tracker's first pass plus your corrections. |  |
| Data | Every flight is logged with timestamps in a machine readable format: camera frames, detections and confidences, re-ID embeddings, distance and angle estimates, velocity commands, flight-mode switches, GPS source switches, and lost-track events. A labeled dataset of crops marked "target" and "not target" is built from that footage and stored on the author's machine, outside the public repository. | Publishing footage or labeled crops that show other people who have not agreed to it. The author's own footage is not excluded: keeping it and using it in the demo video and the report are in scope. The restriction is on what leaves the author's machine, because the training data will include bystanders on public trails | A public, anonymized version of the dataset (faces blurred, bystanders removed) if the build goes public. |
| Metrics | Eight metrics are measured and reported: detection rate (fraction of frames with the person in view where the detector finds them), track continuity (longest and mean unbroken track), reacquisition accuracy (fraction of losses where the correct person is reacquired), false-lock rate (fraction of reacquisitions that lock onto someone else), end-to-end latency (camera frame to velocity command), frames per second on the Jetson, distance error over time against the set station distance, and obstacle-stop response time (first obstacle reading inside the stop distance to the drone reaching zero commanded velocity). | No comparison against published benchmark leaderboards; the metrics are measured on the project's own footage and trails, not on public datasets. |  |
|  |  |  |  |

Everything that does not serve the success criteria is out of scope or deferred to a future phase.

| Area | In scope | Out of scope | Deferred to Phase 2 or later |
| --- | --- | --- | --- |
| Sensing | One forward-facing camera is the only sensor used to find and follow the person. The software on it detects people, tracks the chosen person frame to frame, and re-identifies that same person from any angle after a loss. Short-range time-of-flight sensors on the sides, top, bottom, and rear cover the camera's blind spots for obstacle stops. An optical-flow sensor and a downward rangefinder give position hold without GPS. A GPS module is fitted and logged but nothing in the follow loop or the failsafes depends on it. When GPS drops out or returns mid-flight, the estimator switches sources with no jump in position and no change in behavior, and logs the switch. Occlusion handling is in scope: a trail guarantees the person is briefly hidden behind trees and turns, and recovering the same person afterward is a required behavior. | No depth camera or LiDAR in Phase 1. No tracker or phone carried by the person that reports their position to the drone. No face recognition. Nothing worn or carried by the person for the purpose of being found, visible or not. | A stereo or second camera is a Phase 2 addition on the same airframe. Indoor flight is an optional later phase. |
| Control | The controller holds a set distance and a set angle from the person (2.5 m, behind and above, by default), holds altitude from the downward rangefinder, and orbits a half circle while the person stands still. On an obstacle reading inside the stop distance it hovers, sidesteps, and resumes. On track loss it hovers in place; on prolonged loss it lands. Every distance, angle, timeout, and gain is a named parameter in one config file. | No orbit while the person is walking in v1.0 (it comes after v1.0 as a point release). No aggressive maneuvers: speeds above walking pace, flips, or banked turns. No return-to-home, because it would depend on GPS. | Front-facing travel. Faster subjects (trail running, mountain biking) at 3 to 8 m/s, where the station distance opens to 4 m or more and the obstacle stop to 2 to 3 m. Steep-terrain following from a forward station angle with a slope-aware altitude rule. Cinematic camera moves beyond the standing orbit. |
| Environment | One person is followed. Any outdoor terrain from open ground through hiking trails to rugged ground qualifies, in daylight and light wind (under about 5 m/s). | No night flying, no rain, no snow deeper than 25 mm (1 in), no thick brush the person could walk through but the drone could not, no crowds where more than two or three people are in frame, and no indoor flight. Bench tests indoors with props removed are not flight and are fine. | Cold-weather operation on bare or near-bare ground. Cold cuts pack capacity 20 to 30 percent, so the 15 min target becomes about 11 min. Snow deeper than 25 mm (1 in) is out of scope for every phase: the airframe is not waterproof and the landing gear does not clear deeper snow (decision 20). |
| Autonomy | After a pilot-commanded takeoff and a mode switch on the radio, the drone flies with no pilot input. The pilot's radio link stays armed the whole flight and any stick input overrides the autonomy instantly. On prolonged track loss the drone hovers on optical flow, then lands in place; it never attempts return-to-home. | No autonomous takeoff or routine landing as features: the drone does not start a mission by itself, land on a pad or marker, or decide the hike is over. Those stay pilot-commanded. The failsafe landing on prolonged loss is different: it is the autopilot's built-in LAND mode, a safety behavior, and it is in scope. | Front-follow while the person is walking, and a climb mode for steep terrain. One-button takeoff and a land-here button are candidates for Phase 2 convenience. |
| Platform | One prebuilt long-range 7-inch quadcopter, 7-inch being the maximum size, with a Jetson Orin Nano Super companion computer plugged in. Integration is minimal-solder: thin signal wires to flight-controller pads are acceptable; battery, ESC, and motor leads are never soldered by the builder. | No fixed-wing or VTOL aircraft. No custom frame design or bare-frame build. No second drone. No companion computer larger than the Orin Nano unless a second camera forces it. | Nothing. This stays a one-drone project; Phase 2 adds sensors to the same airframe. |
| Deliverables | A public code repository, a project board, a risk register, a decision log, a flight log, a written report with measured results, and a 2 to 3 minute demo video. | No paper submission and no patent filing. No YouTube build series as a required deliverable (it is optional; see Open decisions). | After v1.0, a cheaper-compute study: quantize and prune the models, measure accuracy and frame rate at each step, and test whether the same behavior runs on a Raspberry Pi 5 with a low-cost accelerator. |

## Phases and milestones

Six stages over five weeks, starting Sunday November 2 and ending Tuesday December 2, at 15 to 20 hours a week. The order puts everything that needs the sky in the first three weekends and everything that can happen on a desk or a bench in parallel or later. Each stage ends with a tagged release in the repository, even when the release is only software. About 20 percent of each week is left unplanned on purpose.

*(Roadmap diagram lives in the living plan doc; see the Phases table below.)*

Sunset is about 4:20 pm in New Hampshire by mid-November, so flying happens on weekends: November 8 and 9, 15 and 16, 22 and 23, and 29 and 30, plus the Thanksgiving days. If the plan slips past December 2, the push-out rule applies: by the end of stage 4 the airframe is proven, the trail footage is banked, GPS-free hold is proven, and station hold is the only sky item left, so everything else moves indoors for the winter and autonomous follow resumes on the first bare-ground weekend in March.

This timeframe is aggressive in order to get the project completed before the weather creates delays.  The project is at a high risk of slippage due to the high likelihood of academic and personal time conflicts.  The mitigation will be to push the trail testing to the first available bare ground windows.  See risk register for more details.

| Phase | Deliverable | Exit criteria (the gate) |
| --- | --- | --- |
| 0 Setup and order (Nov 2 to 4) | Public repository with README, project board, risk register, decision log, and this plan. The final success-criteria numbers written into the README. FAA TRUST certificate and registration done. Every part ordered from in-stock listings only: 7-inch bind-and-fly with an ArduPilot-supported board, Jetson Orin Nano Super kit, USB global-shutter camera, 12 V 5 A BEC with a barrel-jack lead for the Jetson, optical-flow sensor, downward rangefinder, six packs, two-port charger, prop guards, JST-GH cable pack. | Success numbers are yours, not placeholders; every order confirmed with a delivery date inside stage 2. |
| 1 Software (Nov 2 to 11, desk, in parallel) | ArduPilot software-in-the-loop (SITL) running on the Mac with QGroundControl. Perception pipeline (detector, tracker, re-ID, enrollment, logging) running on the Mac with a webcam. Controller (velocity commands, hold on distance and bearing, state machine, watchdog) tested against SITL with a fake target. Config file with every parameter named. When the Jetson arrives: flashed, set up, pipeline ported, TensorRT export done. Release v0.1. | Simulated drone holds station on a fake target in SITL; detector at 15 frames per second or better on the Jetson. |
| 2 Airframe (Nov 5 to 11, sky on Nov 8 and 9) | Drone bound, flown manually on stock firmware, then flashed to ArduPilot, calibrated, failsafes and modes set, hovered in Stabilize and AltHold with the vibration log checked. Release v0.2. | Manual flight with radio override confirmed on ArduPilot. |
| 3 Data and GPS-free hold (Nov 12 to 18, sky on Nov 15 and 16) | Jetson, camera, and regulator mounted and powered; MAVLink link confirmed; optical flow and rangefinder configured; Loiter proven with GPS disabled. Three or four hover-and-walk data sessions on a trail, 60 minutes of footage banked. This is the weather-sensitive stage and it does not move. Release v0.3. | 60 minutes of trail footage on disk with the target in frame; drone holds position on optical flow for two minutes with GPS off. |
| 4 Closed loop (Nov 19 to 25, sky on Nov 22 and 23) | Full loop on the bench with props off, then tethered hover under autonomy, then first untethered station hold and a straight-line walking follow in an open field. First 500 crops labeled from stage 3 footage. Release v0.9. | Drone holds station on the author, untethered, for one minute; flight logs saved. |
| 5 Flight and report (Nov 26 to Dec 2, sky on the Thanksgiving weekend) | Occlusion test, loss-hover-land test, standing orbit, five-minute follow, gain tuning. Written report, 2 to 3 minute demo video, release v1.0, Phase 2 close-out note. | All Phase 1 success criteria met in two separate outdoor sessions; report answers every open question in the decision log; video shows one occlusion recovery. If not met by December 2, the push-out rule engages. |

Task breakdown by stage. Every task is two hours or less of hands-on time. Waiting time (charging, shipping, firmware builds) is not counted. Hours are best-case estimates with the known failure case noted where one is common; log actual against estimate for the first three tasks and scale the rest.

**Stage 0, Setup and order (Nov 2 to 4), about 5.5 h**

- [ ] Create the public repository, README skeleton, folder layout: 0.5 h
- [ ] Project board with the six stages as columns and the first week's tasks as cards: 0.5 h
- [x] Risk register and decision log in `RISKS.md` and `DECISIONS.md`: 0.75 h (done Nov 4)
- [x] Decide and write the final success-criteria numbers into the README: 0.5 to 1 h (done Nov 4)
- [ ] FAA TRUST certificate: 0.5 h. Drone registration on airframe arrival (serial number required): 0.25 h
- [ ] Order a handheld anemometer; wind speed at the launch point is a required flight-log column: 0.25 h
- [ ] Confirm the bind-and-fly's board is on the ArduPilot supported list and count spare UARTs from the wiring diagram: 1 h
- [ ] Place every order from in-stock listings, delivery dates recorded in the decision log: 1 h (done Nov 3; Li-Ion packs by Nov 9 and prop guards after measuring the airframe remain)

**Stage 1, Software (Nov 2 to 11, desk), about 24 to 30 h**

- [ ] Install ArduPilot SITL and QGroundControl on the Mac: 1 h
- [ ] First simulated flight: arming, modes, Loiter, Guided: 1 h
- [ ] Detector running on the Mac with a webcam: 1 h
- [ ] Tracker added, stable IDs confirmed: 1 h
- [ ] Re-ID model installed, embeddings on crops, latency measured: 1.5 h (failure case: dependency install on the Jetson later, add 1 h)
- [ ] Enrollment logic, similarity threshold, lock and loss state machine: 1.5 h
- [ ] Logging of detections, IDs, embeddings, timestamps: 1 h
- [ ] Walk-around test with the webcam, threshold tuning, false-lock count: 1.5 h
- [ ] Velocity command message to Guided mode, type mask and frame understood, tested in SITL: 1 h (failure case: wrong mask silently ignored, add 1 h)
- [ ] Proportional hold on distance and bearing against a fake target in SITL: 1.5 h
- [ ] State machine Enroll, Follow, Lost (hover), Land: 1.5 h
- [ ] Watchdog on command rate, speed limit, geofence, override check: 1 h
- [ ] Config file with every parameter named: 1 h
- [ ] Read the TAKER H743 BT wiring diagram and ArduPilot board page; assign receiver, GPS, Jetson MAVLink and MTF-01 to UARTs; set four TF-Lunas to I2C with distinct addresses over the FTDI adapter; confirm the GEP-M1025 GPS includes a compass and order one if not: 2 h
- [ ] Soldering practice on pads with 30 AWG wire, two sessions: 2 h
- [ ] Jetson, on arrival: flash the JetPack 6 SD image to the Samsung EVO 512 GB from the Mac with balenaEtcher, first boot headless over USB-C or with the DisplayPort adapter, then migrate to the NVMe: 0.5 h (failure case: firmware update needed before boot, add 1 h)
- [ ] First boot, Ubuntu setup, network, SSH, updates: 1 h
- [ ] PyTorch from NVIDIA's wheels for the JetPack version; no source builds: 1 h
- [ ] USB camera stream verified at flight resolution and frame rate: 0.5 h
- [ ] Port the perception code, export the detector to TensorRT, measure frames per second: 1.5 h

**Stage 2, Airframe (Nov 5 to 11, sky on Nov 8 and 9), about 11 to 14 h**

- [ ] Unbox on Nov 5, inspect props and arms, confirm the FC label reads TAKER H743 BT, cell-check and charge the three RDQ 1300 mAh packs: 0.5 h
- [ ] Swap the TBS Nano receiver for the BeeCeiver ELRS diversity receiver, bind, build the radio model, map channels, arm switch: 1.5 h
- [ ] Stock-firmware check and first hover in the yard: 1 h
- [ ] Two manual sessions of 15 to 20 minutes: 2 h
- [ ] Back up the stock config, flash ArduPilot through the bootloader: 1 h
- [ ] Frame type, motor order and direction test in QGroundControl: 1 h
- [ ] Calibrate accelerometer, compass, radio, ESCs: 1.5 h
- [ ] Modes on a switch (Stabilize, AltHold, Loiter, Guided), battery failsafe, radio-loss failsafe to Land, geofence: 1.5 h
- [ ] Hover in Stabilize, then AltHold, vibration log checked: 1.5 h
- [ ] Decide the Remote ID approach (module, ArduPilot OpenDroneID, or none) and record it in the decision log; order hardware if chosen: 0.5 h, due Nov 11
- [ ] Manual PID pass only if it oscillates: 0 to 2 h

**Stage 3, Data and GPS-free hold (Nov 12 to 18, sky on Nov 15 and 16), about 12 to 15 h**

- [ ] Power chain bench test: Y splitter, XT60 pigtail, 24 V to 12 V converter and Wago joints taped and zip-tied; verify 12 V at the barrel jack with the multimeter before it meets the Jetson; full Jetson load, no brownout: 1 h
- [ ] Jetson and camera mounted on standoffs, cables routed: 1.5 h
- [ ] MAVLink link Jetson to flight controller, heartbeat and attitude confirmed, serial parameters set: 1.5 h
- [ ] Optical-flow sensor and rangefinder cabled and parameters set: 1.5 h
- [ ] Power mode, fan curve, thermal check under load: 0.5 h
- [ ] Flow calibration flight and log review: 1.5 h (failure case: scale wrong, repeat once, add 1 h)
- [ ] Loiter test with GPS disabled, two minutes: 1 h
- [ ] Data session 1, hover-and-walk on a trail, three packs: 1.5 h
- [ ] Data sessions 2 and 3, second pilot if available: 3 h in two blocks
- [ ] Handheld rig walk for moving-camera footage: 1 h

**Stage 4, Closed loop (Nov 19 to 25, sky on Nov 22 and 23), about 11 to 14 h**

- [ ] Props-off full loop on the bench: Jetson commands, flight controller in Guided, outputs respond: 1.5 h
- [ ] Label the first 500 crops from stage 3 footage, tracker-assisted: 2 h
- [ ] Fix what the bench finds: 2 h
- [ ] Tethered hover under autonomy with the author as the target, in the barn with vehicles removed and prop guards on, override confirmed: 2 h
- [ ] Fixes from the tether: 1.5 h
- [ ] First untethered station hold in an open field: 1.5 h
- [ ] Walking follow in a straight line: 1.5 h

**Stage 5, Flight and report (Nov 26 to Dec 2), about 11 to 13 h**

- [ ] Occlusion test behind a tree: 1.5 h
- [ ] Loss, hover, land test: 1 h
- [ ] Standing orbit, half circle: 1.5 h
- [ ] Five-minute follow attempt: 1 h
- [ ] Gain tuning and log review between sessions: 2 h in two blocks
- [ ] Report draft: 2 h
- [ ] Demo video cut: 2 h
- [ ] v1.0 tag, release notes, Phase 2 close-out note: 0.5 h

Total: about 75 to 92 hours over five weeks, front-loaded into stages 1 and 2.

## Platform decision

Decided 2026-10-30 (approach) and 2026-11-03 (hardware). Decision log entries 6 through 17.

**Chosen.** A prebuilt bind-and-fly 7-inch long-range quadcopter, the GEPRC MOZ7 V2, with its stock Betaflight firmware replaced by ArduPilot on the TAKER H743 BT flight controller, and an NVIDIA Jetson Orin Nano Super developer kit as the companion computer running perception, estimation, and control. The companion computer sends velocity setpoints to the autopilot over MAVLink (Micro Air Vehicle Link) and reads telemetry back. ArduPilot SITL (software in the loop) on the development machine is used before any flight.

**Why this option.**

| Requirement | How the choice meets it |
| --- | --- |
| Open flight stack with full control of the control loop | ArduPilot is open source and the H743 is an official ArduPilot target. Guided mode accepts velocity setpoints from the companion computer; the autopilot's stabilization, failsafes, and modes are unchanged. |
| Carry a companion computer, camera, and sensors with usable flight time | 7-inch class with a 336 mm wheelbase, 750 g stock, rated for 6S Li-Ion packs; the frame has top-plate room for the Jetson carrier and two battery mounting positions. Estimated 15 to 18 min with payload on 8000 mAh Li-Ion. |
| GPS treated as unavailable | Optical flow and rangefinder give position hold without GPS under ArduPilot; the fitted GEP-M1025 GPS is logged but nothing in the follow loop depends on it. |
| Run detector plus re-ID on one camera at 15 fps or better | Orin Nano Super runs a YOLO-class detector and a re-ID embedding model on one camera stream with TensorRT; 7 to 25 W from a 7 to 20 V barrel-jack input. |
| Minimal soldering | Bind-and-fly arrives with motors, ESC (electronic speed controller), flight controller, GPS, and receiver installed. Builder-added connections are thin signal wires, pre-crimped JST cables, and a zero-solder power chain (XT60 splitter, 24 V to 12 V converter, Wago lever connectors). |
| 7-inch maximum size | 336 mm wheelbase. |
| In stock, domestic shipping, inside the stage 2 window | Ordered Nov 3, delivered Nov 4. |

**Rejected.** A consumer drone with an SDK (software development kit) hides the flight controller and limits offboard control. A micro drone of the Crazyflie class cannot carry a real camera and needs external positioning. Simulation only produces no real-world evidence. The ModalAI Starling 2 Max arrives flying with PX4 and visual-inertial odometry but costs $4,200 to about $6,000 configured, three to four times a build. A Holybro X500 developer kit breaks the 7-inch maximum. A bare-frame 7-inch build requires power soldering outside current skill. Every F722-based 7-inch bind-and-fly (Chimera7 Pro V2, CineLR 7, Rekon7 Pro, Crocodile75 V3, Kolas 7) was excluded because ArduPilot needs 1 MB of flash. The Raspberry Pi 5 was rejected as the companion computer because it cannot run detector plus re-ID at usable rates on CPU alone. Details and prices are in the decision log.

**Accepted issues and risks.**

| Accepted | Consequence | Handling |
| --- | --- | --- |
| The O4 Pro digital video system is unused | About $200 of the $820 price buys nothing the project needs | Can be removed and sold; the project camera is a separate USB global-shutter unit on the Jetson |
| Stock firmware is Betaflight, not ArduPilot | One-time flash through the bootloader, then full calibration and failsafe setup; the vendor's tune does not carry over | Stage 2 tasks; stock configuration backed up first; manual PID pass budgeted at 0 to 2 h |
| The TBS Nano receiver does not match the existing ELRS radio | Receiver swap and bind before first flight | BeeCeiver ELRS diversity receiver ordered with the airframe; 1.5 h stage 2 task |
| The flight controller's onboard 12 V 2.5 A regulator cannot power the Jetson | Separate 12 V 5 A path from the main pack adds mass and a failure point | Bench-tested at full Jetson load before the Jetson flies |
| No Remote ID broadcast | Required for a registered aircraft outside an FRIA | Open decision, due Nov 11 (see decision log entry 19) |
| Li-Ion pack sizing depends on all-up weight that is not yet measured | 10 min criterion fails on endurance if packs underdeliver | Packs ordered by Nov 9 against measured weight; stage 2 flies on existing 1300 mAh LiPos |
| First-time ArduPilot integration on a racing-class frame | Vibration, compass interference from the ESC, and tuning risk are higher than on a Pixhawk-class kit | Vibration log check and compass calibration are explicit stage 2 gates |

Phase 1 flies one camera plus the time-of-flight sensors that cover its blind spots (sides, up, down, and rear). A second camera or stereo module is a Phase 2 addition on the same frame (decision log entry 6).

## Architecture

Four modules in a line (camera, perception, estimation, control) talk to the autopilot, and a safety module can override control at any time. Each module is a separate process with a logged interface, so any one can be replaced without touching the others.

*(Architecture diagram lives in the living plan doc; see the Modules table below.)*

Perception and estimation are highlighted because they are the two modules that carry the AI content of the project (detection, tracking, estimation).

| Module | What it does | Starting choice | Design note |
| --- | --- | --- | --- |
| Perception | Locks onto the person in frame at takeoff, keeps that identity frame to frame, and reacquires the same person after a loss using appearance from any angle | Pretrained person detector plus a multi-object tracker and a pretrained re-ID embedding model to start; from Phase 3, fine-tune the re-ID model (and the detector if needed) on the project's own bench and flight footage, with before-and-after metrics in the report | Define the input as "a target" so the same interface serves walking and future, faster modes |
| Estimation | Turns bounding-box size and position into range and bearing; coasts on the last estimate when the track is lost | Filtering method to be decided with the author before stage 1 estimation work begins; candidates are an exponential smoother and a Kalman filter (open decision) | Estimation is relative only: distance and angle to the person, never absolute position, so GPS loss changes nothing here |
| Control | Converts range and bearing error into velocity commands that hold a set distance at a set station angle; orbit is that angle sweeping; on an obstacle reading, hovers, sidesteps, and resumes | Proportional control with limits; add damping only if it oscillates | Rule-based by design; no learned policy; station distance and stop distance are parameters so faster modes can open them up |
| Safety | Geofence, altitude floor and ceiling, obstacle sensors with a 1.5 m stop rule, hover on track loss beyond the timeout, land in place on prolonged loss, pilot override always armed, prop guards | Autopilot built-in failsafes plus one project watchdog process | GPS-free: position hold on optical flow plus rangefinder; prolonged loss means hover then land, not return-to-home |
| Logging | Records every frame's detection, estimate, command, and event with timestamps | Comma-separated files per flight, plotted after each session | The logs are the training dataset and the report's evidence |

The autopilot owns stabilization and motor control. The companion computer sends velocity setpoints through MAVLink Guided mode and reads telemetry back. The companion computer never commands attitude or motors directly; this boundary is fixed for the life of the project.

**Training loop.** The three user-facing steps (identify the person, recognize the target, follow per parameters) are three kinds of work. Enrollment is a rule plus a pretrained re-ID embedding. Recognition is the trained part. Following is rule-based control. The loop that produces the trained part, repeated from Phase 3 on:

1. Collect bench and flight footage with the logger running.
2. Label crops of the target and of other people, mostly by the tracker, with manual correction.
3. Fine-tune the re-ID model on the labeled set.
4. Evaluate reacquisition accuracy and false-lock rate on held-out footage.
5. Export to the companion computer and measure frame rate and accuracy on the bench.
6. Record the before-and-after numbers in the report.

## Risk register

The risk register is maintained in `RISKS.md` as the single source. Risks are numbered from 1, ordered by likelihood times impact, and kept on the list when accepted, mitigated, or closed. The register is reviewed at every stage gate.

## Decision log

The decision log is maintained in `DECISIONS.md` as the single source. One dated entry per decision with the options considered and the reason. Entries are written by the author.

## Appendix: repository and documentation conventions

The repository is run as a program, so the artifacts of running it (board, risk register, decision log, releases) are deliverables in their own right.

| Item | Convention |
| --- | --- |
| Repository | Public GitHub repository with a project board (columns: Backlog, This week, In progress, Done). Risks and decisions are GitHub Issues so each has a number to cite. JIRA is an acceptable substitute; the board matters, the brand does not. |
| `README.md` | What the project is, the final success-criteria numbers, how to run the simulator, how to run on hardware. |
| `DECISIONS.md` | Single source for decisions. One entry per decision, dated, with the options considered and the reason. |
| `RISKS.md` | Single source for risks. Numbered from 1, ordered by likelihood times impact, reviewed at each gate; accepted, mitigated, and closed risks stay on the list. |
| `FLIGHT_LOG.md` | One row per flight or simulation run: date, build tag, what was tested, result, link to the log file. |
| Releases | A tagged release at each stage gate (v0.1 software, v0.2 airframe, v0.3 data, v0.9 closed loop, v1.0 flight and report) with three to five lines of release notes. A release that only runs in simulation is still a release. |
| Authorship | AI does lookup, summary, critique, and boilerplate. Every design decision, decision-log entry, and sentence of results is written by the author. |
| Cadence | Fixed weekly blocks on the calendar; each block ends with a commit and a one-line board status. Coursework wins ties. Gates stretch; they are not skipped. |
| Close-out | At v1.0, a one-page note listing what Phase 2 can reuse as-is and what it cannot. Phase 2 does not start before that note exists. |
