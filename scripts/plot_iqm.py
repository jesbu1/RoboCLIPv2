"""
Plot IQM + 95% CI for MetaWorld success rate curves.

Usage:
    python scripts/plot_iqm.py

Requires: pip install rliable wandb matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import wandb
from rliable import library as rly
from rliable import metrics

# ============ CONFIG ============
ENTITY = "rewind-valuemodel"
PROJECT = "rewind-policy-training"

ENVS = [
    "window-close-v2",
    "reach-wall-v2",
    "faucet-close-v2",
    "coffee-button-v2",
    "button-press-wall-v2",
    "door-lock-v2",
    "handle-press-side-v2",
    "sweep-into-v2",
]
SEEDS = [42, 32, 0]

# Map method name -> wandb run name pattern
# Pattern: {env}_nochunk_{method}_seed{seed}
METHODS = {
    "baseline": "baseline",
    "diff": "diff",
    "diff_gamma099": "diff_gamma099",
}

EVAL_TIMESTEPS = list(range(10000, 110000, 10000))  # 10k, 20k, ..., 100k
# ================================


def fetch_success_rates(method_key):
    """
    Fetch success rate curves for all envs × seeds for a given method.
    Returns: dict mapping timestep -> np.array of shape (n_seeds, n_tasks)
    """
    api = wandb.Api()

    # scores[seed_idx][task_idx] = list of (timestep, success_rate)
    all_curves = {}

    for seed_idx, seed in enumerate(SEEDS):
        for task_idx, env in enumerate(ENVS):
            run_name = f"{env}_nochunk_{method_key}_seed{seed}"

            # Find the run
            runs = api.runs(
                f"{ENTITY}/{PROJECT}",
                filters={"display_name": run_name},
            )
            runs = list(runs)

            if len(runs) == 0:
                print(f"  [WARNING] Run not found: {run_name}")
                continue

            # Pick the most recently created run
            runs = sorted(runs, key=lambda r: r.created_at, reverse=True)
            run = runs[0]

            # Get success rate history
            history = run.scan_history(keys=["eval/success_rate"], page_size=1000)
            sr_values = []
            for row in history:
                if "eval/success_rate" in row and row["eval/success_rate"] is not None:
                    sr_values.append(row["eval/success_rate"])

            if len(sr_values) == 0:
                print(f"  [WARNING] No success_rate data: {run_name}")
                continue

            # Align to EVAL_TIMESTEPS (eval happens every 10k steps → 10 evals)
            for t_idx, t in enumerate(EVAL_TIMESTEPS):
                if t_idx < len(sr_values):
                    if t not in all_curves:
                        all_curves[t] = np.full((len(SEEDS), len(ENVS)), np.nan)
                    all_curves[t][seed_idx, task_idx] = sr_values[t_idx]

    return all_curves


def compute_iqm_with_ci(scores_dict):
    """
    Compute IQM and 95% CI at each timestep.
    scores_dict: timestep -> np.array (n_seeds, n_tasks)
    Returns: timesteps, iqm_values, ci_lower, ci_upper
    """
    timesteps = sorted(scores_dict.keys())
    iqm_values = []
    ci_lower = []
    ci_upper = []

    for t in timesteps:
        scores = scores_dict[t]  # (n_seeds, n_tasks)

        # Skip if too many NaNs
        if np.isnan(scores).sum() > scores.size * 0.5:
            iqm_values.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
            continue

        # Replace NaN with 0 for computation
        scores_clean = np.nan_to_num(scores, nan=0.0)

        # rliable expects dict: {"method": scores} where scores is (n_runs, n_tasks)
        # For IQM, flatten seeds into runs dimension
        score_dict = {"method": scores_clean}

        # Compute IQM + 95% CI via bootstrap
        iqm_result, iqm_cis = rly.get_interval_estimates(
            score_dict,
            metrics.aggregate_iqm,
            reps=50000,
        )

        iqm_values.append(iqm_result["method"])
        ci_lower.append(iqm_cis["method"][0])
        ci_upper.append(iqm_cis["method"][1])

    return np.array(timesteps), np.array(iqm_values), np.array(ci_lower), np.array(ci_upper)


def compute_per_task_mean_std(scores_dict):
    """
    Compute per-task mean ± std across seeds at each timestep.
    Returns: dict[env] -> (timesteps, means, stds)
    """
    timesteps = sorted(scores_dict.keys())
    per_task = {env: {"mean": [], "std": []} for env in ENVS}

    for t in timesteps:
        scores = scores_dict[t]  # (n_seeds, n_tasks)
        for task_idx, env in enumerate(ENVS):
            vals = scores[:, task_idx]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                per_task[env]["mean"].append(np.mean(vals))
                per_task[env]["std"].append(np.std(vals))
            else:
                per_task[env]["mean"].append(np.nan)
                per_task[env]["std"].append(np.nan)

    for env in ENVS:
        per_task[env]["mean"] = np.array(per_task[env]["mean"])
        per_task[env]["std"] = np.array(per_task[env]["std"])

    return np.array(timesteps), per_task


def plot_iqm(all_method_data, save_path="figures/iqm_success_rate.pdf"):
    """Plot IQM + 95% CI for all methods."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = {"baseline": "C0", "diff": "C1", "diff_gamma099": "C2"}
    labels = {"baseline": "Baseline P(s)", "diff": "Diff γ=1.0", "diff_gamma099": "Diff γ=0.99"}

    for method_name, scores_dict in all_method_data.items():
        timesteps, iqm, ci_lo, ci_hi = compute_iqm_with_ci(scores_dict)

        color = colors.get(method_name, "C3")
        label = labels.get(method_name, method_name)

        ax.plot(timesteps, iqm, label=label, color=color, linewidth=2)
        ax.fill_between(
            timesteps,
            np.clip(ci_lo, 0, 1),
            np.clip(ci_hi, 0, 1),
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Environment Steps", fontsize=13)
    ax.set_ylabel("IQM Success Rate", fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_xlim(EVAL_TIMESTEPS[0], EVAL_TIMESTEPS[-1])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved IQM plot to {save_path}")
    plt.close()


def plot_per_task(all_method_data, save_path="figures/per_task_success_rate.pdf"):
    """Plot per-task mean ± std for all methods."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    colors = {"baseline": "C0", "diff": "C1", "diff_gamma099": "C2"}
    labels = {"baseline": "Baseline P(s)", "diff": "Diff γ=1.0", "diff_gamma099": "Diff γ=0.99"}

    for task_idx, env in enumerate(ENVS):
        ax = axes[task_idx]

        for method_name, scores_dict in all_method_data.items():
            timesteps, per_task = compute_per_task_mean_std(scores_dict)
            mean = per_task[env]["mean"]
            std = per_task[env]["std"]

            color = colors.get(method_name, "C3")
            label = labels.get(method_name, method_name)

            ax.plot(timesteps, mean, label=label, color=color, linewidth=1.5)
            ax.fill_between(
                timesteps,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                alpha=0.2,
                color=color,
            )

        ax.set_title(env, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlim(EVAL_TIMESTEPS[0], EVAL_TIMESTEPS[-1])
        ax.grid(True, alpha=0.3)

        if task_idx == 0:
            ax.legend(fontsize=7)

    # IQM in the last subplot
    ax = axes[len(ENVS)]
    for method_name, scores_dict in all_method_data.items():
        timesteps, iqm, ci_lo, ci_hi = compute_iqm_with_ci(scores_dict)
        color = colors.get(method_name, "C3")
        label = labels.get(method_name, method_name)
        ax.plot(timesteps, iqm, label=label, color=color, linewidth=1.5)
        ax.fill_between(
            timesteps,
            np.clip(ci_lo, 0, 1),
            np.clip(ci_hi, 0, 1),
            alpha=0.2,
            color=color,
        )
    ax.set_title("All Tasks IQM + 95% CI", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_xlim(EVAL_TIMESTEPS[0], EVAL_TIMESTEPS[-1])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved per-task plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)

    all_method_data = {}

    for method_name, method_key in METHODS.items():
        print(f"\nFetching data for: {method_name}")
        scores_dict = fetch_success_rates(method_key)
        if scores_dict:
            all_method_data[method_name] = scores_dict
        else:
            print(f"  No data found for {method_name}, skipping.")

    if all_method_data:
        plot_iqm(all_method_data)
        plot_per_task(all_method_data)
        print("\nDone!")
    else:
        print("No data found for any method.")
