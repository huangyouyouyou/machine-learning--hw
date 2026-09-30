# P5: Perceptron, Pocket, and AI-assisted verification

Complete the three TODOs in `p5.py`:

1. `perceptron_step`: implement one update using a current mistake.
2. `update_pocket`: retain the lowest-training-error model, with independent weight storage; keep the earlier model on ties.
3. `audit_training_history`: numerically check that pocket training error is nonincreasing and equals the best current training error seen so far.

The data loader, feature extraction, random selection, history recording, test evaluation, and plotting are supplied. The current perceptron and the pocket follow the same update trajectory. Do not restart current weights from the pocket or select a model using test error.

## Run

Install the dependencies listed in the parent folder's requirements file. From this directory:

```bash
python p5.py
python p5.py --audit-only
```

The default experiment uses seed 0 and at most 2000 mistake updates; it stops early if there are no current mistakes. Optional flags include `--seed`, `--max-updates`, `--output-dir`, `--train-data`, and `--test-data`. A starter run raises `NotImplementedError` until the TODOs are completed.

## Conventions and evidence

The digit data are `train_data.mat` and `test_data.mat`. Digit 1 is +1 and digit 6 is -1. The two features are mean pixel intensity and mean absolute left-right mirror difference (asymmetry). Inputs are augmented as `[1, intensity, asymmetry]`; initial weights are `[0, 0, 0]`. Scores greater than or equal to zero predict +1; negative scores predict -1.

The deterministic toy uses augmented rows `[[1,0,0], [1,0,0], [1,0,0], [1,1,0]]`, labels `[+1,-1,+1,-1]`, seed 0, and six mistake updates. Identical inputs with opposite labels make it nonseparable. The trace must include the initial state and the six updated states. Use the actual saved parameters and errors to support your explanation.

Full runs save `outputs/error_curves.png`, `decision_boundaries.png`, `trajectory.csv`, `toy_trace.csv`, and `summary.json`. An audit-only run writes `toy_trace.csv` and `summary.json` to the same output directory; use a separate `--output-dir` if you want to preserve the full-run summary. The boundary plot samples 500 training examples per class, or all available examples if fewer.

For the AI collaboration report, state your prediction before inspecting the curves, include a relevant real interaction excerpt, show your verification evidence, and explain your final decision. Explain separately what is guaranteed for pocket training error and what is observed on the test set. You may validate and retain a correct initial AI answer; no fabricated mistake or conversation is required. See P5 in the homework for the full requirements and scoring.
