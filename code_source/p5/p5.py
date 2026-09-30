"""DDA5001 HW1 P5 STUDENT STARTER: complete the three marked TODOs.

Run `python p5.py` for the digit experiment and `python p5.py --audit-only`
for the deterministic toy trace. The test set is evaluated only AFTER training.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_mat(path, d=16):
    import scipy.io

    data = scipy.io.loadmat(path)["zip"]
    return data[:, 1:].reshape(-1, d, d), data[:, 0].astype(int)


def cal_feature_cls(images, labels, cls_A=1, cls_B=6):
    """Return [intensity, asymmetry], with digit 1 -> +1 and digit 6 -> -1."""
    selected = (labels == cls_A) | (labels == cls_B)
    images = images[selected]
    intensity = images.mean(axis=(1, 2))
    asymmetry = np.abs(images - images[:, :, ::-1]).mean(axis=(1, 2))
    features = np.column_stack((intensity, asymmetry))
    y = np.where(labels[selected] == cls_A, 1, -1)
    return features, y


def add_bias(features):
    """The model input is [1, intensity, asymmetry]."""
    return np.column_stack((np.ones(len(features)), features))


def predict(theta, X):
    """Tie convention: score >= 0 predicts +1; otherwise predict -1."""
    return np.where(X @ theta >= 0.0, 1, -1)


def cal_error(theta, X, y):
    return float(np.mean(predict(theta, X) != y))


def perceptron_step(theta, x, y):
    """One update on a misclassified example; return fresh weights."""
    # TODO 1: return the updated current weights using this example.
    # Do not modify theta in place or restart from the pocket weights.
    raise NotImplementedError("TODO 1: implement the perceptron update")


def update_pocket(theta, current_error, pocket, pocket_error):
    """Keep the first best model on ties, and never alias current weights."""
    # TODO 2: return (new_pocket, new_pocket_error), using TRAINING error only.
    # Retain the earlier pocket model on ties, and return independent storage.
    raise NotImplementedError("TODO 2: implement the pocket update")


def train_pocket(X, y, max_updates=2000, seed=0):
    """Generate ONE current-perceptron trajectory and retain its best model.

    Training takes no test data. Each step randomly selects one current mistake.
    The current weights are never restarted from the pocket weights. History
    includes the initial state (update 0), even if no updates are necessary.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) == 0 or y.shape != (len(X),):
        raise ValueError("X must be nonempty (n, d), and y must have shape (n,).")
    if not np.isin(y, [-1, 1]).all() or not np.isfinite(X).all():
        raise ValueError("Features must be finite and labels must be +1 or -1.")
    if max_updates < 0:
        raise ValueError("max_updates must be nonnegative.")
    rng = np.random.default_rng(seed)
    theta = np.zeros(X.shape[1])
    pocket = theta.copy()
    current_error = cal_error(theta, X, y)
    pocket_error = current_error
    history = {key: [] for key in (
        "update", "chosen_index", "current_weights", "pocket_weights",
        "current_train_error", "pocket_train_error")}

    def record(update, chosen_index):
        history["update"].append(update)
        history["chosen_index"].append(chosen_index)
        # Copies are essential: later updates must not rewrite old snapshots.
        history["current_weights"].append(theta.copy())
        history["pocket_weights"].append(pocket.copy())
        history["current_train_error"].append(current_error)
        history["pocket_train_error"].append(pocket_error)

    record(0, -1)
    for update in range(1, max_updates + 1):
        mistakes = np.flatnonzero(predict(theta, X) != y)
        if len(mistakes) == 0:
            break
        chosen = int(rng.choice(mistakes))
        theta = perceptron_step(theta, X[chosen], y[chosen])
        current_error = cal_error(theta, X, y)
        pocket, pocket_error = update_pocket(
            theta, current_error, pocket, pocket_error)
        record(update, chosen)
    return {key: np.asarray(value) for key, value in history.items()}


def audit_training_history(history, X, y):
    """Check guarantees from saved weights and errors, not from plot shapes."""
    current = history["current_train_error"]
    pocket = history["pocket_train_error"]
    # TODO 3: use numerical assertions to check (i) pocket training error is
    # nonincreasing; (ii) at each k it equals the lowest current training error
    # observed in states 0,...,k. Use tolerance 1e-12 for floating-point checks.
    # A plot or a printed 'passed' without checking these claims is not an audit.
    raise NotImplementedError("TODO 3: implement the two training-error checks")
    # Check that reported errors agree with the actual saved parameters.
    for name in ("current", "pocket"):
        recomputed = [cal_error(w, X, y) for w in history[name + "_weights"]]
        np.testing.assert_allclose(
            history[name + "_train_error"], recomputed, atol=1e-12, rtol=0)
    assert np.array_equal(history["update"], np.arange(len(current)))
    assert not np.shares_memory(
        history["current_weights"], history["pocket_weights"])
    return {"pocket_training_nonincreasing": True,
            "pocket_equals_running_best_training_error": True,
            "snapshot_errors_verified": True,
            "separate_current_and_pocket_storage": True}


def evaluate_test_history(history, X_test, y_test):
    """Read frozen snapshots after training; never select models with test data."""
    return {name + "_test_error": np.array([
        cal_error(w, X_test, y_test) for w in history[name + "_weights"]])
        for name in ("current", "pocket")}


def write_trace(path, history, test_errors=None):
    fields = ["update", "chosen_index", "current_train_error", "pocket_train_error"]
    d = history["current_weights"].shape[1]
    weight_fields = [f"{name}_w{j}" for name in ("current", "pocket") for j in range(d)]
    test_errors = {} if test_errors is None else test_errors
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + weight_fields + list(test_errors))
        writer.writeheader()
        for k in range(len(history["update"])):
            row = {key: history[key][k].item() for key in fields}
            row.update({f"{name}_w{j}": history[name + "_weights"][k, j]
                        for name in ("current", "pocket") for j in range(d)})
            row.update({key: value[k] for key, value in test_errors.items()})
            writer.writerow(row)


def run_toy_audit(output_dir):
    """The first two identical inputs have opposite labels: not separable."""
    X = np.array([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    y = np.array([1, -1, 1, -1])
    history = train_pocket(X, y, max_updates=6, seed=0)
    # Preserve the actual trace even when an audit exposes an implementation bug.
    write_trace(Path(output_dir) / "toy_trace.csv", history)
    checks = audit_training_history(history, X, y)
    print("Toy audit (initial state plus six mistake updates):")
    for k in range(len(history["update"])):
        print(f"  k={k}: current={history['current_weights'][k].tolist()}, "
              f"pocket={history['pocket_weights'][k].tolist()}, "
              f"Ein(current)={history['current_train_error'][k]:.2f}, "
              f"Ein(pocket)={history['pocket_train_error'][k]:.2f}")
    return checks


def plot_error_curves(path, history, test_errors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split, errors in zip(axes, ("train", "test"), (history, test_errors)):
        ax.plot(history["update"], errors[f"current_{split}_error"],
                color="tab:orange", linewidth=1, label="Current perceptron")
        ax.plot(history["update"], errors[f"pocket_{split}_error"],
                color="tab:blue", linewidth=1.5, label="Pocket")
        ax.set(title=f"{split.capitalize()} error", xlabel="Number of mistake updates",
               ylabel="Fraction misclassified", ylim=(0, 1))
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_boundary(ax, theta, limits, label, color):
    """Handle sloped, vertical, and constant classifiers without dividing by 0."""
    bias, w1, w2 = theta
    low, high = limits
    if abs(w2) > 1e-12:
        x = np.array([low, high])
        ax.plot(x, -(bias + w1 * x) / w2, color=color, linewidth=2, label=label)
    elif abs(w1) > 1e-12:
        ax.axvline(-bias / w1, color=color, linewidth=2, label=label)
    else:
        prediction = "+1" if bias >= 0 else "-1"
        ax.plot([], [], color=color, label=f"{label}: constant {prediction}")


def plot_boundaries(path, features, y, history, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, digit, color, marker in ((1, "1", "tab:blue", "o"),
                                       (-1, "6", "tab:orange", "x")):
        available = np.flatnonzero(y == label)
        selected = rng.choice(available, size=min(500, len(available)), replace=False)
        ax.scatter(features[selected, 0], features[selected, 1], s=15, alpha=0.5,
                   color=color, marker=marker, label=f"Digit {digit} (train)")
    # Freeze limits before adding boundaries, so a near-vertical line cannot
    # make the training scatter unreadable through automatic axis expansion.
    xmin, xmax = features[:, 0].min(), features[:, 0].max()
    ymin, ymax = features[:, 1].min(), features[:, 1].max()
    xpad, ypad = max(0.05 * (xmax - xmin), .01), max(0.05 * (ymax - ymin), .01)
    limits = (xmin - xpad, xmax + xpad)
    draw_boundary(ax, history["current_weights"][-1], limits, "Final current", "crimson")
    draw_boundary(ax, history["pocket_weights"][-1], limits, "Final pocket", "black")
    ax.set(xlim=limits, ylim=(ymin - ypad, ymax + ypad), xlabel="Intensity",
           ylabel="Asymmetry", title="Training features and final boundaries")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, default=here / "train_data.mat")
    parser.add_argument("--test-data", type=Path, default=here / "test_data.mat")
    parser.add_argument("--output-dir", type=Path, default=here / "outputs")
    parser.add_argument("--max-updates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"toy_audit": run_toy_audit(args.output_dir)}
    if not args.audit_only:
        train_images, train_labels = load_mat(args.train_data)
        features, y = cal_feature_cls(train_images, train_labels)
        X = add_bias(features)
        history = train_pocket(X, y, args.max_updates, args.seed)
        summary["training_audit"] = audit_training_history(history, X, y)

        # Only after training is complete do we load/evaluate the test set.
        test_images, test_labels = load_mat(args.test_data)
        test_features, y_test = cal_feature_cls(test_images, test_labels)
        test_errors = evaluate_test_history(history, add_bias(test_features), y_test)
        write_trace(args.output_dir / "trajectory.csv", history, test_errors)
        plot_error_curves(args.output_dir / "error_curves.png", history, test_errors)
        plot_boundaries(args.output_dir / "decision_boundaries.png", features, y,
                        history, args.seed)
        summary.update({"seed": args.seed, "max_updates": args.max_updates,
                        "updates_completed": int(history["update"][-1]),
                        "n_train": len(y), "n_test": len(y_test),
                        "score_tie_rule": "score >= 0 -> +1; score < 0 -> -1"})
        for name in ("current", "pocket"):
            summary[name] = {
                "final_weights": history[name + "_weights"][-1].tolist(),
                "final_train_error": float(history[name + "_train_error"][-1]),
                "final_test_error": float(test_errors[name + "_test_error"][-1])}
        summary["interpretation"] = (
            "Only pocket TRAINING error is guaranteed nonincreasing. Test error "
            "is measured after training and is not used to select a model.")
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Outputs saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
