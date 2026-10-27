\# Anis Team — More than a Hack 2027: possession reconstruction



One label per frame (`H`, `A`, `D`, `X`) for FC Barcelona – Athletic Club, built from the \*\*tactical

tracking and Smart Tagging only\*\*. Labels come from an algorithm; nothing is annotated by hand. The

public 5-minute ground truth was used only as a sanity check of the pipeline, never to select

parameters.



\## Reproduce



1\. `python -m pip install -r requirements.txt -r requirements-starter.txt`

2\. Place the organisers' data bundle in `data/` (or set `MTAH\_DATA\_DIR`). It is \*\*not\*\* in this repository.

3\. `python src/make\_submission.py --out outputs/submission.csv`

4\. `python evaluation/validate\_submission.py outputs/submission.csv`



The pipeline is deterministic. Expected SHA-256 of the output (tested on Python 3.13.9 with NumPy 2.5.4 and pandas 3.1.0; other versions may change the hash):

`8aef2c5d7c49ef0cb57ef6f4cb575b63fdfd1c9d48da7bd18835645e2f9ea21f`



\## Method in short



1\. \*\*Dead ball first.\*\* A logistic model on collective player motion (left/right time windows) and

&#x20;  ball-visibility signals gives the probability that the ball is dead. Its 18 coefficients are frozen in

&#x20;  `models/dead\_detector.json`.

2\. \*\*Possession inside each alive phase.\*\* A two-state (H/A) Viterbi decoder with a fixed switching cost

&#x20;  combines three evidence sources: the nearest player to the ball (weighted by distance and margin

&#x20;  between the teams), a collective team-shape model trained on automatic pseudo-labels from the same

&#x20;  match's tracking, and Smart Tagging phase priors.

3\. `X` is written outside the graded halves only.



\## External data



The dead-ball detector was trained on two matches of the open IDSSE dataset (J03WPY, J03WOY; CC BY 4.0),

with the tracking degraded to resemble ours:



> Bassek, M., Rein, R., Weber, H., \& Memmert, D. (2025). An integrated dataset of spatiotemporal and

> event data in elite soccer. \*Scientific Data\*, 12(1), 195.



These files are not redistributed here. Retraining is optional and needs them in `data/external/`

(see `experiments/`).



\## Layout



\- `src/make\_submission.py` — the whole pipeline

\- `models/dead\_detector.json` — frozen detector coefficients

\- `experiments/` — diagnostics and benchmarks run on the open matches

\- `evaluation/` — the organisers' scorer, unchanged



Limitations are described in the technical document.

