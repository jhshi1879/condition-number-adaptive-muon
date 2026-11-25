#!/usr/bin/env python3
"""Create a reproducible notebook and factual report from an alpha-sweep run."""

import argparse
import base64
import hashlib
import json
import math
import re
import statistics
import sys
import os
import uuid
from pathlib import Path


STAGES = ("screening", "refinement", "confirmation")
LABELS = {
    "screening": "Round 1",
    "refinement": "Round 2",
    "confirmation": "Round 3",
}
VAL_PATTERN = re.compile(
    r"step:(\d+)/\d+\s+val_loss:([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


class AnalysisError(ValueError):
    """Raised when a sweep summary cannot support trustworthy analysis."""


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError("{} must be numeric".format(name))
    if not math.isfinite(value):
        raise AnalysisError("{} must be finite".format(name))
    return float(value)


def _alpha_key(value):
    return round(float(value), 12)


def _load_summary(path, raw=None):
    try:
        if raw is None:
            raw = path.read_bytes()
        summary = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise AnalysisError("Cannot read {}: {}".format(path, exc))
    if not isinstance(summary, dict):
        raise AnalysisError("summary.json must contain a JSON object")
    status = summary.get("status")
    if status in ("running", "failed"):
        raise AnalysisError(
            "Refusing analysis: sweep status is {!r}; only completed runs are accepted".format(status)
        )
    if status not in ("completed", "completed_with_divergences"):
        raise AnalysisError("Unsupported or missing sweep status: {!r}".format(status))
    return summary


def _validate_config(summary):
    config = summary.get("config")
    if not isinstance(config, dict):
        raise AnalysisError("summary config must be an object")
    required = (
        "alpha_grid", "screen_steps", "refine_steps", "finalist_steps",
        "screening_seed", "finalist_seeds", "top_k", "val_tokens",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise AnalysisError("Missing config fields: {}".format(", ".join(missing)))
    if not isinstance(config["alpha_grid"], list) or not config["alpha_grid"]:
        raise AnalysisError("config.alpha_grid must be a non-empty list")
    config["alpha_grid"] = [_finite_number(value, "config.alpha_grid item")
                            for value in config["alpha_grid"]]
    if len(set(_alpha_key(value) for value in config["alpha_grid"])) != len(config["alpha_grid"]):
        raise AnalysisError("config.alpha_grid contains duplicate values")
    for key in ("screen_steps", "refine_steps", "finalist_steps", "val_tokens", "top_k"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AnalysisError("config.{} must be a positive integer".format(key))
    seeds = config["finalist_seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise AnalysisError("config.finalist_seeds must be a non-empty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise AnalysisError("config.finalist_seeds must contain integers")
    if len(set(seeds)) != len(seeds):
        raise AnalysisError("config.finalist_seeds contains duplicates")
    seed = config["screening_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AnalysisError("config.screening_seed must be an integer")
    return config


def _stage_grid(summary, stage, config):
    grids = summary.get("round_grids")
    grid = grids.get(stage) if isinstance(grids, dict) else None
    if grid is None and stage == "screening":
        grid = config["alpha_grid"]
    if not isinstance(grid, list) or not grid:
        raise AnalysisError("Missing non-empty round_grids.{} list".format(stage))
    values = [_finite_number(value, "{} alpha".format(stage)) for value in grid]
    if len(set(_alpha_key(value) for value in values)) != len(values):
        raise AnalysisError("round_grids.{} contains duplicate alphas".format(stage))
    return sorted(values)


def _read_curve(row, sweep_dir):
    directory = row["directory"]
    path = Path(directory)
    if not path.is_absolute():
        path = sweep_dir / path
    path = path / "console.log"
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError as exc:
        raise AnalysisError("{} alpha={} seed={}: missing console.log".format(
            row["_stage"], row["alpha"], row["seed"])) from exc
    points = {}
    for match in VAL_PATTERN.finditer(text):
        step, loss = int(match.group(1)), float(match.group(2))
        if math.isfinite(loss):
            points[step] = loss
    if not points:
        raise AnalysisError("{} alpha={} seed={}: no validation curve in console.log".format(
            row["_stage"], row["alpha"], row["seed"]))
    return [{"step": step, "val_loss": points[step]} for step in sorted(points)]


def analyze(summary, sweep_dir):
    """Validate sweep data, aggregate complete alpha trials, and extract curves."""
    config = _validate_config(summary)
    warnings = []
    expected = {}
    row_by_key = {}

    for stage in STAGES:
        grid = _stage_grid(summary, stage, config)
        if stage == "screening" and set(_alpha_key(v) for v in grid) != set(
                _alpha_key(v) for v in config["alpha_grid"]):
            raise AnalysisError("round_grids.screening does not match config.alpha_grid")
        seeds = [config["screening_seed"]] if stage != "confirmation" else config["finalist_seeds"]
        expected[stage] = (grid, seeds)
        rows = summary.get(stage)
        if not isinstance(rows, list):
            raise AnalysisError("summary.{} must be a list".format(stage))
        allowed_alphas = set(_alpha_key(alpha) for alpha in grid)
        allowed_seeds = set(seeds)
        stage_steps = {
            "screening": config["screen_steps"],
            "refinement": config["refine_steps"],
            "confirmation": config["finalist_steps"],
        }[stage]
        for row in rows:
            if not isinstance(row, dict):
                raise AnalysisError("Every {} record must be an object".format(stage))
            for key in ("status", "alpha", "seed", "step", "final_val_loss",
                        "wall_seconds", "directory", "logdir"):
                if key not in row:
                    raise AnalysisError("{} record is missing {}".format(stage, key))
            alpha = _finite_number(row["alpha"], "{} alpha".format(stage))
            seed = row["seed"]
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise AnalysisError("{} seed must be an integer".format(stage))
            if _alpha_key(alpha) not in allowed_alphas or seed not in allowed_seeds:
                raise AnalysisError("{} record has unexpected alpha/seed".format(stage))
            if row["status"] not in ("completed", "diverged"):
                raise AnalysisError("{} record has unsupported status {!r}".format(stage, row["status"]))
            key = (stage, _alpha_key(alpha), seed)
            if key in row_by_key:
                raise AnalysisError("Duplicate {} result for alpha={} seed={}".format(stage, alpha, seed))
            normalized = dict(row)
            normalized["alpha"] = alpha
            normalized["_stage"] = stage
            row_by_key[key] = normalized
            step = row["step"]
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise AnalysisError("{} step must be a non-negative integer".format(stage))
            runtime = _finite_number(row["wall_seconds"], "{} wall_seconds".format(stage))
            if runtime < 0:
                raise AnalysisError("{} wall_seconds must be non-negative".format(stage))
            if not isinstance(row["directory"], str) or not row["directory"]:
                raise AnalysisError("{} directory must be a non-empty path".format(stage))
            if not isinstance(row["logdir"], str):
                raise AnalysisError("{} logdir must be a string".format(stage))
            if row["status"] == "completed":
                _finite_number(row.get("final_val_loss"), "{} final_val_loss".format(stage))
                if step != stage_steps:
                    raise AnalysisError(
                        "{} completed result for alpha={} seed={} has step {}; expected {}".format(
                            stage, alpha, seed, step, stage_steps)
                    )

    for stage in STAGES:
        grid, seeds = expected[stage]
        for alpha in grid:
            for seed in seeds:
                if (stage, _alpha_key(alpha), seed) not in row_by_key:
                    raise AnalysisError(
                        "Completed sweep summary is missing required {} result for alpha={} seed={}".format(
                            stage, alpha, seed)
                    )

    round_data = {}
    winners = {}
    counts = {"diverged_records": 0, "missing_records": 0, "excluded_alpha_rounds": 0}
    for stage in STAGES:
        grid, seeds = expected[stage]
        points = []
        coverage = []
        excluded = []
        eligible = []
        for alpha in grid:
            present = []
            diverged = 0
            missing = 0
            for seed in seeds:
                row = row_by_key.get((stage, _alpha_key(alpha), seed))
                if row is None:
                    missing += 1
                elif row["status"] == "diverged":
                    diverged += 1
                else:
                    present.append(row)
            counts["diverged_records"] += diverged
            counts["missing_records"] += missing
            can_aggregate = not missing and not diverged and len(present) == len(seeds)
            coverage.append({
                "alpha": alpha, "required": len(seeds), "completed": len(present),
                "diverged": diverged, "missing": missing, "eligible": can_aggregate,
            })
            if not can_aggregate:
                counts["excluded_alpha_rounds"] += 1
                reasons = []
                if diverged:
                    reasons.append("{} divergent required seed(s)".format(diverged))
                if missing:
                    reasons.append("{} missing required result(s)".format(missing))
                excluded.append({"alpha": alpha, "reason": " and ".join(reasons)})
                warnings.append("{} alpha={}: excluded ({}); expected {} seed result(s), found {} completed".format(
                    stage, alpha, " and ".join(reasons), len(seeds), len(present)))
                continue
            losses = [float(row["final_val_loss"]) for row in present]
            runtimes = [float(row["wall_seconds"]) for row in present if "wall_seconds" in row]
            item = {
                "alpha": alpha,
                "mean": statistics.mean(losses),
                "std": statistics.pstdev(losses),
                "n": len(losses),
                "required": len(seeds),
                "losses": losses,
                "runtime_seconds": sum(runtimes) if len(runtimes) == len(seeds) else None,
            }
            points.append(item)
            eligible.append((item, present))
        round_data[stage] = {
            "grid": grid, "points": points, "coverage": coverage, "excluded": excluded,
        }
        if eligible:
            winner, winner_rows = min(eligible, key=lambda pair: (pair[0]["mean"], pair[0]["alpha"]))
            curves = []
            for row in winner_rows:
                curve = _read_curve(row, sweep_dir)
                final = curve[-1]
                if final["step"] != row["step"]:
                    raise AnalysisError(
                        "{} winner alpha={} seed={}: final console.log step {} does not match result step {}".format(
                            stage, winner["alpha"], row["seed"], final["step"], row["step"])
                    )
                if abs(final["val_loss"] - float(row["final_val_loss"])) > 5.1e-5:
                    raise AnalysisError(
                        "{} winner alpha={} seed={}: final console.log loss does not match result loss".format(
                            stage, winner["alpha"], row["seed"])
                    )
                curves.append({"seed": row["seed"], "points": curve})
            winners[stage] = {
                "alpha": winner["alpha"], "mean": winner["mean"], "curves": curves,
                "expected_seeds": seeds,
            }
        else:
            winners[stage] = {"alpha": None, "mean": None, "curves": [], "expected_seeds": seeds}

    final_points = sorted(round_data["confirmation"]["points"],
                          key=lambda item: (item["mean"], item["alpha"]))
    if not final_points:
        raise AnalysisError("No confirmation alpha has all required seeds completed; no best alpha is reportable")
    best = final_points[0]
    gap = final_points[1]["mean"] - best["mean"] if len(final_points) > 1 else None
    summary_best = summary.get("best_alpha")
    if summary_best is None or _alpha_key(_finite_number(summary_best, "best_alpha")) != _alpha_key(best["alpha"]):
        raise AnalysisError("summary.best_alpha disagrees with the best fully observed confirmation mean")
    ranking = summary.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != len(final_points):
        raise AnalysisError("summary.ranking disagrees with complete confirmation results")
    for ranked, observed in zip(ranking, final_points):
        if not isinstance(ranked, dict):
            raise AnalysisError("summary.ranking contains a malformed entry")
        try:
            ranked_alpha = _finite_number(ranked.get("alpha"), "ranking.alpha")
            ranked_mean = _finite_number(ranked.get("mean_val_loss"), "ranking.mean_val_loss")
            ranked_std = _finite_number(ranked.get("std_val_loss"), "ranking.std_val_loss")
            ranked_losses = ranked.get("seed_losses")
            if not isinstance(ranked_losses, list):
                raise AnalysisError("summary.ranking seed_losses must be a list")
            ranked_losses = [_finite_number(value, "ranking.seed_losses item") for value in ranked_losses]
        except AnalysisError as exc:
            raise AnalysisError("summary.ranking is malformed: {}".format(exc))
        same = (
            _alpha_key(ranked_alpha) == _alpha_key(observed["alpha"])
            and math.isclose(ranked_mean, observed["mean"], rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(ranked_std, observed["std"], rel_tol=1e-12, abs_tol=1e-12)
            and len(ranked_losses) == len(observed["losses"])
            and all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                    for a, b in zip(ranked_losses, observed["losses"]))
        )
        if not same:
            raise AnalysisError("summary.ranking disagrees with complete confirmation results")

    transform = config.get("transform", "")
    floor_match = re.search(r"spectral_floor\s*=\s*([0-9.eE+-]+)", str(transform))
    spectral_floor = float(floor_match.group(1)) if floor_match else None
    has_divergences = counts["diverged_records"] > 0
    if summary["status"] == "completed" and has_divergences:
        warnings.append("run status is completed but trial records contain divergence")
    if summary["status"] == "completed_with_divergences" and not has_divergences:
        warnings.append("run status reports divergences but no divergent trial record was found")

    compact = {
        "config": {
            "objective": config.get("objective", "mean final validation loss across confirmation seeds"),
            "screening_seed": config["screening_seed"],
            "finalist_seeds": config["finalist_seeds"],
            "screen_steps": config["screen_steps"],
            "refine_steps": config["refine_steps"],
            "finalist_steps": config["finalist_steps"],
            "val_tokens": config["val_tokens"],
            "spectral_floor": spectral_floor,
        },
        "rounds": round_data,
        "winners": winners,
        "best": {
            "alpha": best["alpha"], "mean": best["mean"], "std": best["std"],
            "seed_losses": best["losses"], "gap": gap,
            "next_mean": final_points[1]["mean"] if len(final_points) > 1 else None,
        },
        "counts": counts,
        "warnings": warnings,
        "run_status": summary["status"],
    }
    return compact


PLOT_CODE = r'''import matplotlib.pyplot as plt

# DATA is embedded by post_analysis.py so this notebook can be rerun offline.
STAGE_ORDER = ("screening", "refinement", "confirmation")
STAGE_LABELS = {"screening": "Round 1", "refinement": "Round 2",
                "confirmation": "Round 3"}

# 1. Final validation loss against alpha, with across-seed standard deviations.
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, stage in zip(axes, STAGE_ORDER):
    points = DATA["rounds"][stage]["points"]
    if points:
        ax.errorbar([p["alpha"] for p in points], [p["mean"] for p in points],
                    yerr=[p["std"] for p in points], marker="o", capsize=3)
    else:
        ax.text(0.5, 0.5, "No complete alpha results", ha="center", va="center",
                transform=ax.transAxes)
    ax.set(title=STAGE_LABELS[stage], xlabel="Alpha")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("Final validation loss")
fig.suptitle("Per-round final validation loss")
fig.tight_layout()
plt.show()
plt.close(fig)

# 2. Logged validation curves for each round's best complete alpha; absent
# logs are omitted, never interpolated or synthesized.
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, stage in zip(axes, STAGE_ORDER):
    winner = DATA["winners"][stage]
    if winner["alpha"] is None:
        ax.text(0.5, 0.5, "No complete alpha results", ha="center", va="center",
                transform=ax.transAxes)
    else:
        for curve in winner["curves"]:
            pts = curve["points"]
            ax.plot([p["step"] for p in pts], [p["val_loss"] for p in pts],
                    marker=".", label="seed {}".format(curve["seed"]))
        if winner["curves"]:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No logged curve available", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(r"{} winner $\alpha={:g}$".format(STAGE_LABELS[stage], winner["alpha"]))
    ax.set_xlabel("Training step")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("Logged validation loss")
fig.suptitle("Validation curves for each round winner")
fig.tight_layout()
plt.show()
plt.close(fig)

# 3. Total observed runtime per alpha, only when every required seed has a
# completed result and runtime measurement.
fig, ax = plt.subplots(figsize=(7, 4))
for stage in STAGE_ORDER:
    points = [p for p in DATA["rounds"][stage]["points"]
              if p["runtime_seconds"] is not None]
    if points:
        ax.plot([p["alpha"] for p in points],
                [p["runtime_seconds"] for p in points], marker="o",
                label=STAGE_LABELS[stage])
if any(p["runtime_seconds"] is not None for stage in STAGE_ORDER
       for p in DATA["rounds"][stage]["points"]):
    ax.legend()
else:
    ax.text(0.5, 0.5, "No complete runtime measurements", ha="center", va="center",
            transform=ax.transAxes)
ax.set(xlabel="Alpha", ylabel="Total wall time (seconds)",
       title="Runtime versus alpha")
ax.grid(alpha=0.25)
fig.tight_layout()
plt.show()
plt.close(fig)

# 4. Across-seed population standard deviation in the final round.
fig, ax = plt.subplots(figsize=(7, 4))
points = DATA["rounds"]["confirmation"]["points"]
if points:
    ax.plot([p["alpha"] for p in points], [p["std"] for p in points],
            marker="o")
else:
    ax.text(0.5, 0.5, "No complete round 3 results", ha="center", va="center",
            transform=ax.transAxes)
ax.set(xlabel="Alpha", ylabel="Across-seed std. dev. of final validation loss",
       title="Round 3 seed variability")
ax.grid(alpha=0.25)
fig.tight_layout()
plt.show()
plt.close(fig)

# 5. Search coverage: expected grid points and usable complete points.
fig, ax = plt.subplots(figsize=(7, 4))
labels = [STAGE_LABELS[s] for s in STAGE_ORDER]
expected = [len(DATA["rounds"][s]["coverage"]) for s in STAGE_ORDER]
eligible = [sum(1 for c in DATA["rounds"][s]["coverage"] if c["eligible"])
            for s in STAGE_ORDER]
excluded = [n - ok for n, ok in zip(expected, eligible)]
x = list(range(len(labels)))
ax.bar(x, eligible, label="Complete / included")
ax.bar(x, excluded, bottom=eligible, label="Divergent or missing")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set(ylabel="Alpha grid points", title="Search coverage by round")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
plt.show()
plt.close(fig)
'''


def _notebook_source(data):
    return "DATA = " + repr(data) + "\n\n" + PLOT_CODE


def _render_notebook(data, report_text):
    """Run the notebook plotting cell once and return standard nbformat JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    code = _notebook_source(data)
    rendered = []
    original_show = plt.show

    def capture_show(*args, **kwargs):
        figure = plt.gcf()
        from io import BytesIO
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        rendered.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

    namespace = {"__name__": "__main__"}
    try:
        plt.show = capture_show
        exec(compile(code, "<analysis-notebook-cell>", "exec"), namespace)
    finally:
        plt.show = original_show
        plt.close("all")
    outputs = [{
        "output_type": "display_data",
        "data": {"image/png": png},
        "metadata": {},
    } for png in rendered]
    if len(outputs) != 5:
        raise AnalysisError("Expected five rendered notebook figures, got {}".format(len(outputs)))
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Alpha sweep post-analysis\n",
                    "\n",
                    "This notebook embeds the observed data and standalone plotting code. "
                    "Rerun the code cell to reproduce the figures."
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": report_text.splitlines(True),
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": outputs,
                "source": code.splitlines(True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3",
            },
            "language_info": {"name": "python", "version": "{}.{}.{}".format(
                sys.version_info[0], sys.version_info[1], sys.version_info[2])},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def _report(data):
    config, best, counts = data["config"], data["best"], data["counts"]
    gap_text = ("the next-best complete tested alpha was {:.6g} higher".format(best["gap"])
                if best["gap"] is not None else "no second complete alpha was available for a gap")
    floor_text = (" The configured spectral floor was {}.".format(config["spectral_floor"])
                  if config["spectral_floor"] is not None else "")
    paragraph_one = (
        "The sweep selected $\\alpha={:.12g}$ by minimizing the mean final validation loss, using "
        "objective '{}' and round 3 seeds {}; its observed mean was {:.6g} with "
        "population standard deviation {:.6g}, and {}. Budgets were {} round 1 steps and "
        "{} round 2 steps for seed {}, then {} round 3 steps; each run "
        "used {} validation tokens.{}".format(
            best["alpha"], config["objective"].replace("confirmation", "round 3"),
            config["finalist_seeds"], best["mean"],
            best["std"], gap_text, config["screen_steps"], config["refine_steps"],
            config["screening_seed"], config["finalist_steps"], config["val_tokens"], floor_text,
        )
    )
    warning_text = (" Logged validation curves missing: {}.".format(
        sum(1 for warning in data["warnings"] if "curve" in warning or "console.log" in warning))
                    if any("curve" in warning or "console.log" in warning for warning in data["warnings"])
                    else "")
    paragraph_two = (
        "Run status was {}. Across all rounds, {} required trial record(s) diverged, {} "
        "required result(s) were missing, and {} alpha-round point(s) were excluded; "
        "excluded points are not plotted or used in aggregates. Results are limited to the "
        "tested grid and available seeds, budgets differ by round, and the best tested alpha "
        "is not evidence of a global optimum.{}".format(
            data["run_status"], counts["diverged_records"], counts["missing_records"],
            counts["excluded_alpha_rounds"], warning_text,
        )
    )
    return paragraph_one + "\n\n" + paragraph_two + "\n"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_manifest(manifest_path, summary_sha256, destination):
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise AnalysisError("Cannot validate existing analysis manifest: {}".format(exc))
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise AnalysisError("Existing analysis manifest is not a completed publication")
    if manifest.get("summary_sha256") != summary_sha256:
        raise AnalysisError("Refusing to overwrite analysis outputs for a different summary.json")
    expected_names = ("analysis.ipynb", "experiment_report.md")
    files = manifest.get("files")
    if (not isinstance(files, list) or
            any(not isinstance(item, dict) for item in files) or
            [item.get("name") for item in files] != list(expected_names)):
        raise AnalysisError("Existing analysis manifest has an unexpected publication file list")
    for item in files:
        path = destination / item["name"]
        if not path.is_file():
            raise AnalysisError("Existing analysis output is missing: {}".format(item["name"]))
        if (not isinstance(item.get("size"), int) or
                path.stat().st_size != item["size"] or _sha256(path) != item.get("sha256")):
            raise AnalysisError("Existing analysis output failed manifest integrity check: {}".format(item["name"]))
    return manifest


def _write_manifest_atomic(path, manifest):
    temporary = path.with_name(".{}.tmp.{}.{}".format(path.name, os.getpid(), uuid.uuid4().hex))
    try:
        with temporary.open("x") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def create_analysis(sweep_dir, output_dir=None):
    sweep_dir = Path(sweep_dir).expanduser().resolve()
    summary_path = sweep_dir / "summary.json"
    try:
        summary_bytes = summary_path.read_bytes()
    except OSError as exc:
        raise AnalysisError("Cannot read {}: {}".format(summary_path, exc))
    summary = _load_summary(summary_path, summary_bytes)
    data = analyze(summary, sweep_dir)
    destination = Path(output_dir).expanduser().resolve() if output_dir else sweep_dir / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    manifest_path = destination / "analysis_manifest.json"
    if manifest_path.exists():
        _existing_manifest(manifest_path, summary_sha256, destination)
        return data, destination
    publication_names = ("analysis.ipynb", "experiment_report.md")
    if any((destination / name).exists() for name in publication_names):
        raise AnalysisError("Refusing to overwrite existing analysis outputs without a matching manifest")
    report_text = _report(data)
    notebook = _render_notebook(data, report_text)
    (destination / publication_names[0]).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    (destination / publication_names[1]).write_text(report_text)
    manifest = {
        "status": "completed",
        "summary_sha256": summary_sha256,
        "files": [
            {"name": name, "sha256": _sha256(destination / name),
             "size": (destination / name).stat().st_size}
            for name in publication_names
        ],
    }
    _write_manifest_atomic(manifest_path, manifest)
    return data, destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True,
                        help="completed sweep directory containing summary.json")
    parser.add_argument("--output-dir", help="output directory (default: <sweep-dir>/analysis)")
    args = parser.parse_args(argv)
    try:
        data, destination = create_analysis(args.sweep_dir, args.output_dir)
    except AnalysisError as exc:
        parser.exit(2, "post-analysis: {}\n".format(exc))
    print("Wrote {} and {}".format(destination / "analysis.ipynb",
                                   destination / "experiment_report.md"))
    for warning in data["warnings"]:
        print("WARNING: {}".format(warning), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
