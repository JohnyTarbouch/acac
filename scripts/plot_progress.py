import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-acac")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training curves from an ACAC progress.csv file."
    )
    parser.add_argument(
        "--progress_csv",
        type=Path,
        required=True,
        help="Path to the progress.csv file produced by training.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults next to progress.csv.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    return parser.parse_args()


def maybe_float(value):
    if value is None or value == "":
        return math.nan
    return float(value)


def load_progress(progress_csv):
    with progress_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {progress_csv}")

    data = {}
    for key in reader.fieldnames:
        data[key] = [maybe_float(row.get(key)) for row in rows]
    return data


def build_default_output(progress_csv):
    return progress_csv.with_name("training_result.png")


def infer_title(progress_csv, explicit_title):
    if explicit_title:
        return explicit_title

    parts = progress_csv.parts
    try:
        run_name = parts[-2]
        env_name = parts[-3]
        exp_name = parts[-4]
        return f"{exp_name} / {env_name} / {run_name}"
    except IndexError:
        return progress_csv.stem


def add_stop_marker(ax, x_values, y_values, label):
    if not x_values or not y_values:
        return
    x_last = x_values[-1]
    y_last = y_values[-1]
    ax.scatter([x_last], [y_last], color="#d62728", s=28, zorder=5)
    ax.annotate(
        label,
        xy=(x_last, y_last),
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=9,
        color="#d62728",
    )


def main():
    args = parse_args()
    progress_csv = args.progress_csv.resolve()
    output = (args.output or build_default_output(progress_csv)).resolve()

    data = load_progress(progress_csv)

    timesteps = data["Timesteps"]
    episodes = data["Episodes"]

    eval_return = data["Eval/ReturnAverage"]
    train_return = data["Train/ReturnAverage"]
    eval_epilen = data["Eval/EpiLenAverage"]
    train_epilen = data["Train/EpiLenAverage"]
    epsilon = data["Eps"]
    critic_loss = data["Joint/CriticLoss"]

    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    title = infer_title(progress_csv, args.title)
    final_timestep = timesteps[-1]
    final_episode = episodes[-1]
    fig.suptitle(
        f"{title}\nStopped at {final_timestep:,.0f} timesteps, {final_episode:,.0f} episodes",
        fontsize=14,
        y=0.98,
    )

    axes[0].plot(timesteps, eval_return, label="Eval Return", color="#1f77b4", linewidth=2.2)
    axes[0].plot(
        timesteps,
        train_return,
        label="Train Return",
        color="#ff7f0e",
        linewidth=1.5,
        alpha=0.85,
    )
    axes[0].axvline(final_timestep, color="#d62728", linestyle="--", linewidth=1, alpha=0.7)
    add_stop_marker(axes[0], timesteps, eval_return, "stop")
    axes[0].set_ylabel("Return")
    axes[0].set_title("Return vs Timesteps")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="lower right")

    axes[1].plot(
        timesteps,
        eval_epilen,
        label="Eval Episode Length",
        color="#2ca02c",
        linewidth=2.0,
    )
    axes[1].plot(
        timesteps,
        train_epilen,
        label="Train Episode Length",
        color="#9467bd",
        linewidth=1.5,
        alpha=0.85,
    )
    axes[1].axvline(final_timestep, color="#d62728", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_ylabel("Episode Length")
    axes[1].set_title("Episode Length vs Timesteps")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    axes[2].plot(timesteps, epsilon, label="Epsilon", color="#8c564b", linewidth=1.8)
    axes[2].plot(
        timesteps,
        critic_loss,
        label="Critic Loss",
        color="#e377c2",
        linewidth=1.3,
        alpha=0.9,
    )
    axes[2].axvline(final_timestep, color="#d62728", linestyle="--", linewidth=1, alpha=0.7)
    axes[2].set_xlabel("Timesteps")
    axes[2].set_ylabel("Value")
    axes[2].set_title("Optimization State at Stop")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(output)


if __name__ == "__main__":
    main()
