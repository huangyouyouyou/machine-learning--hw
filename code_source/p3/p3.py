"""P3 student starter: robust regression and evidence-based gradient checks.

Complete the four TODO functions, with coding-agent assistance if useful.
The course objective is the SUM of the shifted Huber functions in the handout.
Run this file from any directory. The default run reproduces the HW1 baseline.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def training_arrays(X, y):
    """Use vectors of shape (n,) so (n, 1) data cannot broadcast to (n, n)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("X must have shape (n, d) and y must contain n values.")
    return X, y


def least_squares(X, y):
    """Return the stable least-squares estimate as a vector of shape (d,).

    TODO 1: use np.linalg.lstsq; do not form an inverse of X.T @ X.
    The supplied data y has shape (n, 1); use training_arrays to flatten it.
    """
    raise NotImplementedError("Implement least_squares (TODO 1).")


def huber_loss(theta, X, y, mu):
    """Return the scalar SUM loss L(theta) exactly as defined in P3(b).

    TODO 2: implement the two branches of h_mu and sum over residuals.
    Check the additive constant carefully: the handout has h_mu(0) = mu/2.
    Flatten theta and y; mu must be positive. Do not take a sample mean.
    """
    raise NotImplementedError("Implement huber_loss (TODO 2).")


def huber_gradient(theta, X, y, mu):
    """Return the gradient of the SUM loss as a vector of shape (d,).

    TODO 3: apply your P3(b) derivative and the chain rule. Check the inner,
    negative outer, and positive outer branches and both boundary values.
    Flatten theta and y; mu must be positive. Do not divide by n.
    """
    raise NotImplementedError("Implement huber_gradient (TODO 3).")


def gradient_checks(X, y, theta, mu, direction, steps):
    """Return a list of dictionaries, one for each positive step t.

    TODO 4: normalize direction to unit length, then compare the actual loss
    change L(theta+t*v)-L(theta) with t * gradient(theta).T @ v.
    Also compare the CENTRAL finite difference of the SUM loss with
    gradient(theta).T @ v. Separately compute the central finite difference of
    the MEAN loss L(theta)/n and compare it with gradient(theta).T @ v / n.
    Record the ratio of the sum-loss and mean-loss finite differences, using
    None if the mean-loss finite difference has magnitude at most 1e-12.
    Finally, check the deliberate mismatch between the sum-loss finite
    difference and the divided-by-n gradient. Do not hard-code expected results.

    Required dictionary keys (used by supplied CSV and summary scaffolding):
      mu, step, step_over_mu,
      actual_loss_change, linear_prediction,
      taylor_abs_error, taylor_abs_error_over_step,
      directional_finite_difference, gradient_dot_direction,
      finite_difference_abs_error, mean_gradient_dot_direction,
      mean_directional_finite_difference, mean_finite_difference_abs_error,
      sum_to_mean_finite_difference_ratio,
      mean_scaling_abs_error.

    All *_abs_error fields are absolute differences. The Taylor error is
    abs(actual_loss_change - linear_prediction). The last field compares
    the finite difference of the SUM loss with the divided-by-n candidate.
    """
    raise NotImplementedError("Implement gradient_checks (TODO 4).")


def gradient_descent(X, y, theta0, mu=1e-5, alpha=0.001, steps=1000):
    """Exactly T updates, with theta/loss histories at k = 0, ..., T.

    theta_star is intentionally not an argument: it is only an evaluation
    reference and cannot influence fitting or the stopping rule.
    """
    X, y = training_arrays(X, y)
    if steps < 0 or alpha <= 0:
        raise ValueError("steps must be nonnegative and alpha must be positive.")
    theta = np.asarray(theta0, dtype=float).reshape(-1).copy()
    theta_history = np.empty((steps + 1, theta.size))
    loss_history = np.empty(steps + 1)
    for k in range(steps + 1):
        theta_history[k] = theta
        loss_history[k] = huber_loss(theta, X, y, mu)
        if k < steps:
            theta = theta - alpha * huber_gradient(theta, X, y, mu)
    return theta, theta_history, loss_history


def write_csv(path, rows):
    """Supplied reporting scaffolding; rows must have consistent keys."""
    if not rows:
        raise ValueError("Cannot write an empty table.")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_verification(output_dir):
    """A deterministic small problem with inner and both outer branches.

    theta = 0 and y = O(mu) avoid cancellation from subtracting large fitted
    values. Steps range from mu to 1e-4*mu, not arbitrarily tiny numbers.
    For the smallest steps no residual crosses a Huber branch boundary.
    """
    X = np.array([[1., 0.], [0., 1.], [1., 1.], [1., -1.], [1., 2.]])
    theta = np.zeros(2)
    direction = np.array([3., 4.]) / 5
    multipliers = np.array([-2., -0.5, 0., 0.5, 2.])
    step_multipliers = np.array([1., 0.3, 0.1, 0.03, 0.01, 0.003,
                                 0.001, 0.0003, 0.0001])
    all_rows = []
    branch_rows = []
    summaries = []
    for mu in (1.0, 1e-5):
        y = -mu * multipliers
        rows = gradient_checks(X, y, theta, mu, direction,
                               mu * step_multipliers)
        all_rows.extend(rows)
        for multiplier in multipliers:
            residual = float(mu * multiplier)
            branch_rows.append({
                "mu": mu,
                "residual_over_mu": float(multiplier),
                "residual": residual,
                "h_mu": huber_loss([residual], [[1.]], [0.], mu),
                "h_mu_derivative": float(
                    huber_gradient([residual], [[1.]], [0.], mu)[0]),
            })
        smallest = rows[-1]
        summaries.append({
            "mu": mu,
            "n": int(X.shape[0]),
            "gradient": huber_gradient(theta, X, y, mu).tolist(),
            "gradient_dot_direction": smallest["gradient_dot_direction"],
            "finite_difference_at_smallest_step":
                smallest["directional_finite_difference"],
            "finite_difference_abs_error_at_smallest_step":
                smallest["finite_difference_abs_error"],
            "error_if_sum_gradient_is_divided_by_n":
                smallest["mean_scaling_abs_error"],
            "sum_to_mean_gradient_factor": int(X.shape[0]),
            "mean_finite_difference_at_smallest_step":
                smallest["mean_directional_finite_difference"],
            "mean_finite_difference_abs_error_at_smallest_step":
                smallest["mean_finite_difference_abs_error"],
            "sum_to_mean_finite_difference_ratio":
                smallest["sum_to_mean_finite_difference_ratio"],
            "taylor_abs_error_at_largest_step": rows[0]["taylor_abs_error"],
            "taylor_abs_error_at_smallest_step": smallest["taylor_abs_error"],
        })
    write_csv(output_dir / "gradient_verification.csv", all_rows)
    write_csv(output_dir / "huber_branches.csv", branch_rows)
    return summaries


def run_baseline(data_dir, output_dir, mu, alpha, steps, seed):
    X, y = training_arrays(np.load(data_dir / "X.npy"),
                           np.load(data_dir / "y.npy"))
    theta0 = np.random.default_rng(seed).standard_normal(X.shape[1])
    theta_ls = least_squares(X, y)
    theta, theta_history, loss_history = gradient_descent(
        X, y, theta0, mu=mu, alpha=alpha, steps=steps)

    # Load ground truth only AFTER fitting; never use it to choose updates.
    theta_star = np.load(data_dir / "theta_star.npy").reshape(-1)
    errors = np.linalg.norm(theta_history - theta_star, axis=1)
    ls_error = float(np.linalg.norm(theta_ls - theta_star))
    rows = [{"iteration": k, "sum_huber_loss": float(loss_history[k]),
             "estimation_error": float(errors[k])} for k in range(steps + 1)]
    write_csv(output_dir / "regression_history.csv", rows)
    np.save(output_dir / "theta_huber.npy", theta)
    np.save(output_dir / "theta_ls.npy", theta_ls)

    # Plotting is provided so the TODOs focus on ML and validation.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(np.arange(steps + 1), errors, label="Huber gradient descent")
    ax.axhline(ls_error, linestyle="--", color="tab:orange",
               label="Least squares")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Iteration k (k = 0 is the initialization)")
    ax.set_ylabel(r"Estimation error $\|\theta_k-\theta^\star\|_2$")
    ax.set_title("P3: robust regression baseline")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output_dir / "regression_error.png", dpi=180)
    plt.close(fig)
    return {
        "n": int(X.shape[0]), "d": int(X.shape[1]), "mu": mu,
        "alpha": alpha, "updates": steps, "history_samples": steps + 1,
        "seed": seed, "objective_reduction": "sum",
        "least_squares_error": ls_error,
        "initial_huber_error": float(errors[0]),
        "final_huber_error": float(errors[-1]),
        "initial_huber_loss": float(loss_history[0]),
        "final_huber_loss": float(loss_history[-1]),
    }


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=here / "data")
    parser.add_argument("--output-dir", type=Path, default=here / "outputs")
    parser.add_argument("--mu", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = {"verification": run_verification(args.output_dir)}
        if not args.verify_only:
            summary["baseline"] = run_baseline(
                args.data_dir, args.output_dir, args.mu, args.alpha,
                args.steps, args.seed)
    except NotImplementedError as error:
        parser.exit(1, f"TODO incomplete: {error}\n")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
