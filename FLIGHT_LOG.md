# Flight log

One row per flight or simulation run. Build tag is the git tag or short commit hash. Log file links point to `logs/` (gitignored) or to the ArduPilot .bin dataflash log kept on the author's machine.

| Date | Build | Type | Site | Wind (m/s) | Pack | What was tested | Result | Log |
|------|-------|------|------|------------|------|-----------------|--------|-----|
| | | sim / bench / tethered / free | | anemometer at launch | | | | |
| 2026-11-06 | ardupilot fix-fwversion | sim | SITL default (Canberra) | n/a | n/a | SITL and QGC install; arm, Guided takeoff to 10 m, Loiter hold, Land | Pass. First Loiter attempt crashed because the simulated throttle was at minimum; passed after `rc 3 1500`. | none |
