# Risk register

Single source for project risks. Numbered from 1 and ordered by likelihood times impact (Low = 1, Medium = 2, High = 3). Risks stay on the list when accepted, mitigated, or closed; the Status column records which. Reviewed at every stage gate. Each risk is also a GitHub Issue labeled `risk`.

| # | Risk | Likelihood | Impact | Score | Mitigation | Trigger to act | Status |
|---|------|------------|--------|-------|------------|----------------|--------|
| 1 | Coursework or job search crowds out build time | High | Medium | 6 | Fixed weekly block for the project; coursework always wins ties; gates stretch, they do not drop; trail testing pushes to the first available bare-ground window if the schedule slips | Two consecutive weeks with no commit | Open |
| 2 | Detector too slow on the companion computer for stable control | Medium | High | 6 | Measure frames per second in stage 1 before any flight; export to TensorRT at INT8 before dropping resolution; smaller model next; the Orin NX module is the last resort and drops into the same carrier | Under 10 frames per second end to end | Open |
| 3 | Crash destroys the airframe, Jetson, or camera | Medium | High | 6 | Tethered flights first; prop guards; low altitude; one spare set of arms and props budgeted | First free flight | Open |
| 4 | Range estimate from a bounding box is too noisy to hold distance | Medium | Medium | 4 | Filter the estimate; widen the distance band; add a known-size reference such as a vest or marker as a fallback | Distance error over 1.5 m for more than 2 s in sim | Open |
| 5 | Winter weather blocks outdoor flight for months | High | Low | 3 | Schedule already puts sim and bench in winter; find an indoor space for tethered tests; push-out rule: by end of stage 4 the airframe is proven, footage banked, GPS-free hold proven, and autonomous follow resumes on the first bare-ground weekend in March | Stage 4 not reached by Nov 25 | Open |
