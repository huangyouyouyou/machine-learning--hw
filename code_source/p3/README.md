# P3: robust regression with gradient verification

Complete four functions in `p3.py`: `least_squares`, `huber_loss`,
`huber_gradient`, and `gradient_checks`. Coding agents are allowed. Follow the
handout for the mathematical answers and the genuine AI collaboration record.

The starter already supplies loading, vector-shape normalization, the gradient
descent loop, random initialization, CSV export, and plotting. Your work should
focus on the estimator, the exact objective, its gradient, and evidence that
the implementation agrees with the mathematics.

## Run

With NumPy and Matplotlib installed, run these commands in this directory:

```sh
python p3.py --verify-only
python p3.py
```

The default run uses `mu=1e-5`, `alpha=0.001`, `T=1000`, and seed `0` with
`np.random.default_rng(0).standard_normal(d)`. It performs exactly 1000 updates
and records iterations 0 through 1000. Data paths are relative to the script,
so it also works when launched from another directory. `--output-dir` changes
where the results are saved; `--help` lists the other options. Use the defaults
for the required baseline and clearly label any optional extra experiments.

## Verify the mathematical object you actually implemented

- The handout uses a **sum**, not a mean, of the shifted Huber functions.
  Its value at residual zero is `mu/2`.
- The supplied verification problem tests residuals
  `mu * [-2, -0.5, 0, 0.5, 2]`, covering the inner branch and both outer branches.
  Run it at both `mu=1` and `mu=1e-5`.
- For several steps, compare the actual loss change with the first-order Taylor
  prediction and compare the central finite difference with the directional
  gradient. Explain what the results establish. A single numerical direction
  is evidence, not a proof for every possible input.
- Separately check finite differences of the sum loss and the mean loss against
  their respective gradients and report the ratio. The verification also
  compares the sum-loss finite difference with a gradient divided by the sample
  count. Explain this mismatch and its consequence for a fixed learning rate.

Do not use `theta_star.npy` in fitting or to select an iteration. The supplied
code loads it only after fitting to evaluate estimation error.

## Outputs

`--verify-only` writes `outputs/gradient_verification.csv`,
`outputs/huber_branches.csv`, and `outputs/summary.json`. The full run also writes
`outputs/regression_error.png`, `outputs/regression_history.csv`,
`outputs/theta_huber.npy`, and `outputs/theta_ls.npy`.

Use the actual outputs as evidence for your P3 response. Include the selected
real prompt/reply excerpts, your own check, and your resulting decision in the
AI collaboration case required by the handout. There is no requirement that
the agent make a mistake or that you change initially correct code.
