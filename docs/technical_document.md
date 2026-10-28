# Anis Team: technical document

More than a Hack 2027, possession reconstruction for FC Barcelona – Athletic Club
Repository: https://github.com/anisabdali/mtah2027-possession

## 1. Summary

We label every frame `H`, `A`, `D` or `X` from the **tactical tracking and the Smart Tagging only**. The pipeline is one deterministic script (`src/make_submission.py`). It works in two stages:

1. decide **when the ball is dead**, from the collective motion of the players and a few ball signals;
2. inside each alive phase, decide **which team is in possession**, by combining three weak signals with a decoder that makes switching possession costly.

We chose this structure because of what the data showed. The ball is missing on about 41% of graded frames, almost all in long gaps (median 10.7 s). A frame-by-frame rule such as "nearest player" therefore fails twice: it writes `D` whenever the ball is missing, and it flips team at every duel. Detecting dead ball first, then enforcing persistent possession, addressed both.

## 2. Assets

**Used**
- Tactical tracking (ATD files): player and ball positions, and the team of each track.
- Smart Tagging: phase codes with a team tag, used as a weak prior on the possessor.
- Two matches of the open IDSSE dataset (J03WPY and J03WOY), used only to train the dead-ball detector and to benchmark the pipeline (see section 4).
- The organisers' scorer (`evaluation/`), unchanged.

**Not used in the final pipeline:** the tactical and broadcast videos, the homography, the broadcast tracking and tagging. We did not try to improve the ball track from video.

**Tried and dropped**
- *Ball missing means dead* (the baseline's rule). The two are strongly related but not equal.
- *A first decoder with a high switching cost and one nearest-player signal.* Possessions lasted 100 s on average. Raising the evidence weight brought back the fragmentation of the raw baseline (1.0 s at the 10th percentile). No single weight gave both stability and reactivity.
- *A logistic regression on geometric features* (nearest and second-nearest distances, players within a radius, ball position), trained on one IDSSE match and tested on the other. It did not beat the plain nearest player (balanced accuracy 79.4% for nearest, 76.8% for the regression at threshold 0.5, 79.7% with a validation-tuned threshold).
- *Re-calibrating the nearest-player weights on IDSSE possession instead of on the tags.* No consistent gain (+0.01 to +0.03 `spellF1` with fine possessions, −0.02 to −0.03 with merged ones), so we kept the original table.

## 3. Method and possession rules

**Stage 1: dead ball.** A logistic regression with 18 features gives the probability that the ball is dead: median speed, share of slow players and 75th-percentile speed over a long centred window and over short windows looking *left* and *right* of each frame (to localise the start and end of a stoppage), plus ball visibility rate, ball speed when visible and share of time outside the pitch. The probability is smoothed over 1 s and thresholded at 0.5. Stoppages shorter than 3 s are ignored, and alive phases shorter than 3 s between two stoppages are absorbed. The 18 coefficients are frozen in `models/dead_detector.json`. They were trained on the two IDSSE matches with their tracking degraded to resemble ours (ball masked by a two-state process, 7% of players removed in 2 s blocks).

**Stage 2: possession in each alive phase.** A two-state Viterbi decoder (`H`/`A`) with a fixed cost for switching, set from a mean possession of about 14 s. Three evidence sources are summed per frame:
- *Nearest player.* A weight indexed by the distance to the nearest player and by the margin between the nearest home and away player (reliable when a player is within 1 m and the opponents are far, uninformative otherwise).
- *Team shape.* A logistic regression on seven collective features (relative depth and width of the two teams, centroid velocity). It is trained for each half on automatic pseudo-labels (frames where the ball is within 1 m of one player and at least 2 m from the nearest opponent). It does not use the ball at inference time, so it keeps working in gaps.
- *Smart Tagging.* Phase codes whose team tag names the attacking team (`BUILD UP`, `PROGRESSION`, `ATTACKING TRANSITION`, `BALL IN FINAL THIRD`, `LONG BALL`) or the defending team (`DEFENDING IN ATTACKING/MIDDLE THIRD`, `DEFENSIVE TRANSITION`, in which case the possessor is the opponent). We found that these codes come in mirror pairs, so we use one signal per frame, from the most reliable covering phase.

**Rules implemented**
- While the ball is alive, exactly one team is in possession. There is no neutral state.
- *Loose balls, single touches, duels, deflections:* possession changes only after consistent evidence over several seconds, so a brief opposing touch does not switch it. We do not model "control" explicitly.
- *Restarts:* each alive phase is decoded on its own, so the possessor after a stoppage does not depend on the one before it.
- *Dead ball is never a possession* and is written `D`. We never predict the owner of a dead ball.
- `X` is written only outside the two graded halves (taken from the organisers' `games.py`, which `possession_metrics.valid_runs` also documents).

## 4. Validation

No private ground truth was used and nothing was annotated by hand. We could only measure on 17 public possessions, so we built a benchmark on the two IDSSE matches.

**4.1 Public window (sanity check).** The final file scores `macroF1` 0.751 and `spellF1` 0.386 on the public five minutes. Team accuracy on frames alive in both is 0.914. Dead-ball precision is 0.86 but recall only 0.43. It predicts 12 possessions for 17 real ones (mean 22.6 s against 14.2 s). With 17 possessions this is not a result we rely on: a published baseline (`baseline_smooth125`) reaches 0.424 `spellF1` on the same window, and the repository warns that this window overstates it.

**4.2 IDSSE benchmark** (scored with the official `evaluate`, tracking degraded as above, no Smart Tagging because IDSSE has none, detector trained on the *other* match). The `spellF1` values are for J03WOY / J03WPY. About ±0.04 is the width of the 5–95% bootstrap intervals over possessions.

| Possession definition | Baseline-like (best window) | Dead-ball detector v1 | **Final detector (v2)** | Perfect dead ball |
|---|---|---|---|---|
| Fine (median 3.6–3.8 s) | 0.224 / 0.247 | 0.333 / 0.339 | **0.380 / 0.417** | 0.459 / 0.472 |
| Spells under 6 s merged (mean 17.9–19.1 s) | 0.206 / 0.308 | 0.444 / 0.467 | **0.514 / 0.570** | 0.615 / 0.676 |

IDSSE possessions are finer than the competition's, so only comparisons between variants are meaningful, not the absolute values.

**4.3 Dead-ball detector.** Balanced accuracy 0.93 / 0.95 on the two matches (0.86 for v1). Half of the stoppage starts are placed within 1 s of the truth (58% and 61%, against 12% and 22% for v1); ends are less precise (44% and 47%). A variant without ball signals reached only 0.87–0.88, so most of the gain comes from them. Stoppages under 10 s are the weakest case (45% and 86% detected).

**4.4 Visual check.** We watched the video at the longest predicted stoppages (2 to 4 minutes each). They correspond to real interruptions, including a hydration break in the first half. This check was not used to produce any label.

**Disclosure.** We ran the public scorer on every version to check the pipeline. We chose the final version on the IDSSE benchmark, with the rule fixed beforehand that only gains shown on both matches count.

## 5. Limitations

- **The detector transfers imperfectly.** Recall on the public window (0.43) is far below its IDSSE recall. Our guess is that our tracking is noisier (identity swaps, missing players) than our degradation simulates, which makes players look faster. We did not test this.
- **Stoppage boundaries** are the main remaining loss: a perfect dead ball would add 0.08 to 0.20 `spellF1` on IDSSE.
- **Short stoppages** (touches, free kicks under 10 s) are often missed.
- **Control is not modelled.** The decoder cannot tell a controlled reception from a rebound, which the challenge itself calls ambiguous. Possession durations are set by a prior (14 s), not learned.
- **Hand-set constants:** the evidence weights, the 14 s prior and the 3 s thresholds were fixed by us. On IDSSE, results were insensitive to the evidence weight (6 to 25) and to the minimum stoppage (3 to 10 s).
- **Tags double-count risk:** the tag reliabilities come from their agreement with the nearest-player estimate, and the tags were produced from the same tracking.
- **No abstention.** We never use `X` on graded frames.
- **Little external data:** two matches, with synthetic degradation, and no way to validate the Smart Tagging part outside the public window.

## 6. With more time

Train on all seven IDSSE matches; degrade them to match measured noise statistics of our tracking; learn a ball-control event detector rather than a persistence prior; use the video or homography to recover the ball in gaps; add the set-piece windows as evidence for the detector.

## 7. Reproduction

```
python -m pip install -r requirements.txt -r requirements-starter.txt
# put the organisers' data bundle in data/, or set MTAH_DATA_DIR
python src/make_submission.py --out outputs/submission.csv
python evaluation/validate_submission.py outputs/submission.csv
```

The output is deterministic (no randomness). Expected SHA-256: `8aef2c5d7c49ef0cb57ef6f4cb575b63fdfd1c9d48da7bd18835645e2f9ea21f`, obtained from a fresh clone of the repository, a new virtual environment, Python 3.13.9, NumPy 2.5.4 and pandas 3.1.0. Other versions may change it. Reading the tracking takes 1 to 3 minutes the first time. Experiments and benchmarks are in `experiments/`. Retraining the detector (optional) needs the IDSSE files in `data/external/`.

## 8. Data and compliance

The labels come from an algorithm. We did not use the private ground truth. The match data is not in the repository. The dead-ball detector uses the open IDSSE data (CC BY 4.0):

Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data*, 12(1), 195.
