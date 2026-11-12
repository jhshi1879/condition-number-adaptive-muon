"""Plot the training and validation losses recorded by the three runs."""

import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RECORD = HERE.parent
FILES = {
    "AdamW": RECORD / "runpod-results/20261108-071147-91210-14344/training.log",
    "Muon": RECORD / "muon-full/training.log",
    "ConditionalNumberAdaptiveMuon": (
        RECORD / "runpod-results/20261108-110947-67696-13584/training.log"
    ),
}
COLORS = {
    "AdamW": "tab:blue",
    "Muon": "tab:purple",
    "ConditionalNumberAdaptiveMuon": "tab:orange",
}
LOG_LINE = re.compile(
    r"^step:(\d+)/\d+ (train_loss|val_loss):"
    r"([0-9.eE+-]+) train_time:(\d+)ms step_avg:([0-9.eE+-]+|nan)ms"
)


@dataclass
class Run:
    train: list
    val: list
    step_ms: float


def read_run(path):
    train = []
    val = []
    step_ms = None
    with path.open(encoding="utf-8") as log:
        for line in log:
            match = LOG_LINE.match(line)
            if not match:
                continue
            step, kind, loss, time_ms, average = match.groups()
            # The logged clock is reset at step 10; earlier times are not comparable.
            if int(step) <= 10:
                continue
            point = (int(time_ms) / 60000, float(loss))
            (train if kind == "train_loss" else val).append(point)
            if average != "nan":
                step_ms = float(average)
    if not train or not val or step_ms is None:
        raise ValueError(f"Missing training, validation, or timing data in {path}")
    return Run(train, val, step_ms)


def smooth(points, window=50):
    """Return a trailing mean of training loss at each logged time."""
    recent = deque()
    total = 0.0
    result = []
    for time_minutes, loss in points:
        recent.append(loss)
        total += loss
        if len(recent) > window:
            total -= recent.popleft()
        result.append((time_minutes, total / len(recent)))
    return result


def create_plot(runs, kind, output):
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, run in runs.items():
        points = smooth(run.train) if kind == "train" else run.val
        times, losses = zip(*points)
        ax.plot(
            times,
            losses,
            color=COLORS[name],
            linewidth=1.8,
            marker="." if kind == "val" else None,
            markersize=4,
            label=f"{name}  ({run.step_ms:.0f} ms/step)",
        )
    ax.set_title(f"Optimizer comparison by time: {kind} loss")
    ax.set_xlabel("Logged training time (minutes)")
    ax.set_ylabel(f"{kind.capitalize()} loss")
    ax.set_ylim(3.0, 4.2)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    note = (
        "Loss outside 3.0-4.2 is clipped. Timing excludes the first 10 steps, "
        "validation, and checkpointing."
    )
    if kind == "train":
        note += " Training loss: trailing 50-step mean."
    fig.text(0.5, 0.015, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")


def create_train_loss_vs_time(runs, name=HERE / "train_loss_vs_time.png"):
    create_plot(runs, "train", name)


def create_val_loss_vs_time(runs, name=HERE / "val_loss_vs_time.png"):
    create_plot(runs, "val", name)


def main():
    runs = {name: read_run(path) for name, path in FILES.items()}
    create_train_loss_vs_time(runs)
    create_val_loss_vs_time(runs)


if __name__ == "__main__":
    main()
