#!/usr/bin/env bash
# One-time GitHub setup. Run from the repository root on the Mac:
#   cd "/Volumes/Primary External Drive/DEV/Drone-Followme" && bash scripts/github_setup.sh
# Needs: gh authenticated as Naroot0 with the `project` scope.
#   gh auth refresh -s project,read:project
set -euo pipefail

OWNER="Naroot0"
REPO="follow-me-drone"
DESC="Vision-only follow-me camera drone for hiking trails. ArduPilot + Jetson Orin Nano. Rule-based control, GPS treated as unavailable."

echo "== issue templates into .github (the folder cannot be written remotely)"
mkdir -p .github/ISSUE_TEMPLATE
cp scripts/issue_templates/*.md .github/ISSUE_TEMPLATE/

echo "== git init and first commit"
if [ ! -d .git ]; then
  git init -b main
fi
git add -A
git commit -m "Stage 0: README with final success criteria, plan, risk register, decision log" || true

echo "== create public repository"
if ! gh repo view "$OWNER/$REPO" >/dev/null 2>&1; then
  gh repo create "$OWNER/$REPO" --public --description "$DESC" --source . --remote origin --push
else
  git remote add origin "git@github.com:$OWNER/$REPO.git" 2>/dev/null || true
  git push -u origin main
fi

echo "== labels"
for L in "risk:B60205:Risk register entry" "decision:0E8A16:Decision log entry" "task:1D76DB:Task of 2 h or less" \
         "stage-0:EDEDED:Setup and order" "stage-1:EDEDED:Software" "stage-2:EDEDED:Airframe" \
         "stage-3:EDEDED:Data and GPS-free hold" "stage-4:EDEDED:Closed loop" "stage-5:EDEDED:Flight and report"; do
  IFS=: read -r name color desc <<<"$L"
  gh label create "$name" --color "$color" --description "$desc" --repo "$OWNER/$REPO" --force
done

echo "== project board"
PROJECT_NUM=$(gh project list --owner "$OWNER" --format json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(next((p["number"] for p in d.get("projects",[]) if p["title"]=="Follow-Me Drone"), ""))')
if [ -z "$PROJECT_NUM" ]; then
  PROJECT_NUM=$(gh project create --owner "$OWNER" --title "Follow-Me Drone" --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["number"])')
fi
echo "Project number: $PROJECT_NUM"
gh project link "$PROJECT_NUM" --owner "$OWNER" --repo "$OWNER/$REPO" 2>/dev/null || true

# Status field: the default single-select field is "Status" with Todo / In Progress / Done.
# Rename options to Backlog / This week / In progress / Done.
FIELD_ID=$(gh project field-list "$PROJECT_NUM" --owner "$OWNER" --format json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(next(f["id"] for f in d["fields"] if f["name"]=="Status"))')
PROJECT_ID=$(gh project view "$PROJECT_NUM" --owner "$OWNER" --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
gh api graphql -f query='
mutation($project:ID!,$field:ID!){
  updateProjectV2Field(input:{projectId:$project, fieldId:$field, name:"Status",
    singleSelectOptions:[
      {name:"Backlog",     color:GRAY,   description:"Not scheduled"},
      {name:"This week",   color:BLUE,   description:"Committed for the current week"},
      {name:"In progress", color:YELLOW, description:"Started"},
      {name:"Done",        color:GREEN,  description:"Finished and committed"}
    ]}){ projectV2Field { ... on ProjectV2SingleSelectField { id } } }
}' -f project="$PROJECT_ID" -f field="$FIELD_ID" >/dev/null

echo "== issues: risks"
python3 - "$OWNER/$REPO" <<'EOF'
import re, subprocess, sys
repo = sys.argv[1]
def rows(path):
    out = []
    for line in open(path):
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| #"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        out.append(cells)
    return out
for c in rows("RISKS.md"):
    n, risk, lik, imp, score, mit, trig, status = c[:8]
    body = f"**Likelihood** {lik}\n\n**Impact** {imp}\n\n**Score** {score}\n\n**Mitigation** {mit}\n\n**Trigger to act** {trig}\n\n_RISKS.md row {n}_"
    subprocess.run(["gh","issue","create","--repo",repo,"--title",f"Risk {n}: {risk}","--body",body,"--label","risk"],check=True)
for c in rows("DECISIONS.md"):
    n, date, dec, opts, reason = c[:5]
    body = f"**Date** {date}\n\n**Options considered** {opts}\n\n**Reason** {reason}\n\n_DECISIONS.md row {n}_"
    title = f"Decision {n}: {dec}"[:200]
    subprocess.run(["gh","issue","create","--repo",repo,"--title",title,"--body",body,"--label","decision"],check=True)
EOF

echo "== issues: first week's tasks (stage 0 remainder, stage 1 start, stage 2)"
while IFS='|' read -r stage title hours; do
  [ -z "$title" ] && continue
  gh issue create --repo "$OWNER/$REPO" --title "$title" --label "task,$stage" \
    --body "**Stage** ${stage#stage-}
**Estimate** ${hours} h (happy path)
**Done when** see docs/PLAN.md task breakdown" >/dev/null
  echo "  + $title"
done <<'EOF'
stage-2|Decide Remote ID approach (module, ArduPilot OpenDroneID, or none) by Nov 11; record and order|0.5
stage-0|Register aircraft in FAADroneZone on arrival (serial number required); mark the frame|0.5
stage-0|Order handheld anemometer (wind is a required flight-log column)|0.25
stage-0|Order 6S Li-Ion packs after measuring all-up weight (by Nov 9)|0.5
stage-0|Order prop guards after measuring the airframe|0.5
stage-0|Confirm TAKER H743 BT on ArduPilot supported list; count spare UARTs|1
stage-1|Install ArduPilot SITL and QGroundControl on the Mac|1
stage-1|First simulated flight: arming, modes, Loiter, Guided|1
stage-1|Detector running on the Mac with a webcam|1
stage-1|Tracker added, stable IDs confirmed|1
stage-1|Re-ID model installed, embeddings on crops, latency measured|1.5
stage-1|Enrollment logic, similarity threshold, lock and loss state machine|1.5
stage-1|Logging of detections, IDs, embeddings, timestamps|1
stage-1|Velocity command message to Guided mode, tested in SITL|1
stage-1|Proportional hold on distance and bearing against a fake target in SITL|1.5
stage-1|State machine Enroll, Follow, Lost (hover), Land|1.5
stage-1|Watchdog on command rate, speed limit, geofence, override check|1
stage-1|Read TAKER H743 BT wiring diagram; assign UARTs; TF-Lunas on I2C; confirm GPS compass|2
stage-1|Soldering practice on pads with 30 AWG wire, two sessions|2
stage-2|Unbox Nov 5, inspect, confirm FC label, cell-check and charge 1300 mAh packs|0.5
stage-2|Swap TBS Nano for BeeCeiver ELRS, bind, radio model, arm switch|1.5
stage-2|Stock-firmware check and first hover in the yard|1
stage-2|Two manual sessions of 15 to 20 minutes|2
stage-2|Back up stock config, flash ArduPilot through the bootloader|1
stage-2|Frame type, motor order and direction test in QGroundControl|1
stage-2|Calibrate accelerometer, compass, radio, ESCs|1.5
stage-2|Modes on a switch, battery failsafe, radio-loss to Land, geofence|1.5
stage-2|Hover in Stabilize then AltHold, vibration log checked|1.5
EOF

echo "== add all issues to the board"
gh issue list --repo "$OWNER/$REPO" --limit 200 --json url --jq '.[].url' | while read -r url; do
  gh project item-add "$PROJECT_NUM" --owner "$OWNER" --url "$url" >/dev/null
done

echo
echo "Done. Repo: https://github.com/$OWNER/$REPO"
echo "Board: https://github.com/users/$OWNER/projects/$PROJECT_NUM"
echo "Next: in the board UI, switch to Board layout, then drag this week's task cards from Backlog to 'This week'."
