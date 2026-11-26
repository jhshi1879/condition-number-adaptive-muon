import base64
import hashlib
import importlib
import io
import json
import shutil
import sys
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SWEEP_CODE = ROOT / "alpha_sweep"
sys.path.insert(0, str(SWEEP_CODE))
import post_analysis


class PostAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "tests" / (".post-analysis-" + uuid.uuid4().hex)
        self.sweep_dir = self.root / "sweep"
        self.sweep_dir.mkdir(parents=True)
        self.stage_steps = {"screening": 10, "refinement": 20, "confirmation": 50}
        self.summary = {
            "status": "completed_with_divergences",
            "config": {
                "alpha_grid": [0.0, 1.0],
                "screen_steps": 10,
                "refine_steps": 20,
                "finalist_steps": 50,
                "screening_seed": 0,
                "finalist_seeds": [0, 1],
                "top_k": 2,
                "val_tokens": 1048576,
                "objective": "mean final validation loss across confirmation seeds",
                "transform": "exact regularized Gram eigendecomposition; spectral_floor=0.01",
            },
            "round_grids": {
                "screening": [0.0, 1.0],
                "refinement": [0.0, 0.5, 1.0],
                "confirmation": [0.5, 1.0],
            },
            "screening": [
                self._row("screening", 0.0, 0, "completed", 5.0, 1.0, log=True),
                self._row("screening", 1.0, 0, "diverged", None, 0.3),
            ],
            "refinement": [
                self._row("refinement", 0.0, 0, "completed", 4.2, 1.0, log=True),
                self._row("refinement", 0.5, 0, "completed", 4.0, 1.1, log=True),
                self._row("refinement", 1.0, 0, "completed", 4.5, 1.2, log=True),
            ],
            "confirmation": [
                self._row("confirmation", 0.5, 0, "completed", 3.0, 2.0, log=True),
                self._row("confirmation", 0.5, 1, "completed", 3.2, 2.1, log=True),
                self._row("confirmation", 1.0, 0, "completed", 2.8, 2.2, log=True),
                self._row("confirmation", 1.0, 1, "diverged", None, 0.5),
            ],
            "ranking": [{
                "alpha": 0.5, "mean_val_loss": 3.1,
                "std_val_loss": 0.1, "seed_losses": [3.0, 3.2],
            }],
            "best_alpha": 0.5,
        }
        self._write_summary()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _row(self, stage, alpha, seed, status, loss, runtime, log=False):
        directory = self.sweep_dir / "{}-{}-{}".format(stage, alpha, seed)
        directory.mkdir(parents=True, exist_ok=True)
        steps = self.stage_steps[stage]
        if log:
            (directory / "console.log").write_text(
                "step:{}/{} val_loss:{:.4f}\nstep:{}/{} val_loss:{:.4f}\n".format(
                    max(1, steps // 2), steps, (loss or 0) + 1,
                    steps, steps, loss or 0,
                )
            )
        return {
            "status": status, "alpha": alpha, "seed": seed,
            "step": steps, "final_val_loss": loss, "wall_seconds": runtime,
            "directory": str(directory), "logdir": str(directory / "tensorboard"),
        }

    def _write_summary(self):
        (self.sweep_dir / "summary.json").write_text(json.dumps(self.summary))

    def _five_rounds(self):
        self.summary["config"]["rounds"] = 5
        self.summary["config"]["objective"] = "mean final validation loss across final-round seeds"
        self.stage_steps["round_5"] = 50
        grids = self.summary["round_grids"]
        self.summary["round_1"] = self.summary.pop("screening")
        self.summary["round_2"] = self.summary.pop("refinement")
        self.summary["round_5"] = self.summary.pop("confirmation")
        self.summary["round_grids"] = {
            "round_1": grids["screening"], "round_2": grids["refinement"],
            "round_3": [0.5], "round_4": [0.5],
            "round_5": grids["confirmation"],
        }
        for stage in ("round_3", "round_4"):
            self.stage_steps[stage] = 20
            self.summary[stage] = [self._row(stage, 0.5, 0, "completed", 3.8, 1, log=True)]
        self._write_summary()

    def _short_sweep(self, count):
        grids = self.summary["round_grids"]
        screening = self.summary.pop("screening")
        self.summary.pop("refinement")
        self.summary.pop("confirmation")
        self.summary["config"]["rounds"] = count
        self.summary["config"]["finalist_steps"] = 500
        self.summary["config"]["objective"] = "mean final validation loss across final-round seeds"
        final_stage = "round_{}".format(count)
        self.stage_steps[final_stage] = 500
        if count == 1:
            self.summary["config"]["alpha_grid"] = grids["confirmation"]
            self.summary["round_grids"] = {final_stage: grids["confirmation"]}
        else:
            self.summary["round_grids"] = {
                "round_1": grids["screening"], final_stage: grids["confirmation"],
            }
            self.summary["round_1"] = screening
        self.summary[final_stage] = [
            self._row(final_stage, 0.5, 0, "completed", 3.0, 2, log=True),
            self._row(final_stage, 0.5, 1, "completed", 3.2, 2, log=True),
            self._row(final_stage, 1.0, 0, "completed", 2.8, 2, log=True),
            self._row(final_stage, 1.0, 1, "diverged", None, 0.5),
        ]
        self._write_summary()
        return final_stage

    def test_analysis_excludes_bad_alpha_and_renders_reproducible_notebook(self):
        data, destination = post_analysis.create_analysis(self.sweep_dir)
        self.assertEqual(data["best"]["alpha"], 0.5)
        self.assertAlmostEqual(data["best"]["mean"], 3.1)
        self.assertAlmostEqual(data["best"]["std"], 0.1)
        self.assertIsNone(data["best"]["gap"])
        self.assertEqual(data["counts"]["diverged_records"], 2)
        self.assertEqual(data["counts"]["missing_records"], 0)
        self.assertEqual(data["counts"]["excluded_alpha_rounds"], 2)
        self.assertEqual([point["alpha"] for point in data["rounds"]["confirmation"]["points"]], [0.5])
        self.assertEqual(data["rounds"]["confirmation"]["points"][0]["losses"], [3.0, 3.2])
        self.assertEqual(data["winners"]["confirmation"]["alpha"], 0.5)
        self.assertEqual(len(data["winners"]["confirmation"]["curves"]), 2)
        self.assertEqual(len(data["winners"]["screening"]["curves"]), 1)

        report = (destination / "experiment_report.md").read_text().strip()
        self.assertEqual(len(report.split("\n\n")), 2)
        self.assertIn("objective 'mean final validation loss across round 3 seeds'", report)
        self.assertIn("round 3 seeds [0, 1]", report)
        self.assertIn("$\\alpha=", report)
        self.assertNotRegex(report, r"(?i)screening|refinement|confirmation")
        self.assertIn("spectral floor was 0.01", report)
        self.assertIn("not evidence of a global optimum", report)

        notebook = json.loads((destination / "analysis.ipynb").read_text())
        self.assertEqual(notebook["nbformat_minor"], 4)
        setup = "".join(notebook["cells"][0]["source"])
        self.assertIn("## Experimental setup", setup)
        self.assertIn("## Alpha and the selection criterion", setup)
        self.assertIn("$\\alpha$", setup)
        self.assertIn("$\\epsilon=0.01$", setup)
        self.assertIn("Round 3: 50 steps", setup)
        self.assertEqual("".join(notebook["cells"][1]["source"]), report + "\n")
        cell = notebook["cells"][2]
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(len(cell["outputs"]), 5)
        self.assertNotIn(str(self.sweep_dir), "".join(cell["source"]))
        for output in cell["outputs"]:
            image = base64.b64decode(output["data"]["image/png"])
            self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))

        manifest_path = destination / "analysis_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(
            [item["name"] for item in manifest["files"]],
            ["analysis.ipynb", "experiment_report.md"],
        )
        self.assertEqual(
            manifest["summary_sha256"],
            hashlib.sha256((self.sweep_dir / "summary.json").read_bytes()).hexdigest(),
        )
        for item in manifest["files"]:
            output_path = destination / item["name"]
            self.assertEqual(item["size"], output_path.stat().st_size)
            self.assertEqual(item["sha256"], hashlib.sha256(output_path.read_bytes()).hexdigest())

        # Verify the embedded cell is independently rerunnable and generates
        # all five figures from its embedded data.
        import matplotlib.pyplot as plt
        shown = []
        original_show = plt.show
        try:
            plt.show = lambda *args, **kwargs: shown.append(plt.gcf())
            exec("".join(cell["source"]), {})
        finally:
            plt.show = original_show
            plt.close("all")
        self.assertEqual(len(shown), 5)
        self.assertEqual([axis.get_title() for axis in shown[0].axes],
                         ["Round 1", "Round 2", "Round 3"])
        self.assertEqual(shown[3].axes[0].get_title(), "Round 3 seed variability")
        for figure in shown:
            for axis in figure.axes:
                self.assertNotRegex(axis.get_title(), r"(?i)screening|refinement|confirmation")
                legend = axis.get_legend()
                if legend is not None:
                    for label in legend.get_texts():
                        self.assertNotRegex(label.get_text(), r"(?i)screening|refinement|confirmation")

    def test_five_rounds_are_validated_and_rendered(self):
        self._five_rounds()
        self.summary["round_3"][0]["status"] = "diverged"
        self.summary["round_3"][0]["final_val_loss"] = None
        self._write_summary()
        data, destination = post_analysis.create_analysis(self.sweep_dir)
        self.assertEqual(data["stage_order"], ["round_{}".format(i) for i in range(1, 6)])
        self.assertEqual(data["best"]["alpha"], 0.5)
        self.assertEqual(len(data["winners"]["round_5"]["curves"]), 2)
        report = (destination / "experiment_report.md").read_text().strip()
        self.assertEqual(len(report.split("\n\n")), 2)
        self.assertIn("round 5 seeds [0, 1]", report)
        self.assertIn("round 4: 20 steps", report)
        notebook = json.loads((destination / "analysis.ipynb").read_text())
        setup = "".join(notebook["cells"][0]["source"])
        self.assertIn("Round 5: 50 steps", setup)
        self.assertIn("$\\bar{L}", setup)
        cell = notebook["cells"][2]
        self.assertEqual(len(cell["outputs"]), 5)
        for output in cell["outputs"]:
            self.assertTrue(base64.b64decode(output["data"]["image/png"]).startswith(b"\x89PNG\r\n\x1a\n"))
        import matplotlib.pyplot as plt
        shown = []
        original_show = plt.show
        try:
            plt.show = lambda *args, **kwargs: shown.append(plt.gcf())
            exec("".join(cell["source"]), {})
        finally:
            plt.show = original_show
            plt.close("all")
        self.assertEqual(len(shown), 5)
        self.assertEqual([ax.get_title() for ax in shown[0].axes],
                         ["Round {}".format(i) for i in range(1, 6)])
        self.assertTrue(all(ax.get_title().startswith("Round {} winner".format(i))
                            for i, ax in enumerate(shown[1].axes, 1)))
        self.assertEqual(shown[3].axes[0].get_title(), "Round 5 seed variability")
        self.assertEqual([label.get_text() for label in shown[4].axes[0].get_xticklabels()],
                         ["Round {}".format(i) for i in range(1, 6)])

    def test_five_rounds_reject_missing_rounds_results_wrong_budgets_and_ranking(self):
        self._five_rounds()
        del self.summary["round_grids"]["round_3"]
        with self.assertRaisesRegex(post_analysis.AnalysisError, "round_grids.round_3"):
            post_analysis.analyze(self.summary, self.sweep_dir)
        self.summary["round_grids"]["round_3"] = [0.5]
        self.summary["round_4"] = []
        with self.assertRaisesRegex(post_analysis.AnalysisError, "missing required round_4 result"):
            post_analysis.analyze(self.summary, self.sweep_dir)
        self.summary["round_4"] = [self._row("round_4", 0.5, 0, "completed", 3.8, 1, log=True)]
        self.summary["round_4"][0]["step"] = 10
        with self.assertRaisesRegex(post_analysis.AnalysisError, "expected 20"):
            post_analysis.analyze(self.summary, self.sweep_dir)
        self.summary["round_4"][0]["step"] = 20
        self.summary["round_5"].pop(0)
        with self.assertRaisesRegex(post_analysis.AnalysisError, "missing required round_5 result"):
            post_analysis.analyze(self.summary, self.sweep_dir)
        self.summary["round_5"].insert(0, self._row("round_5", 0.5, 0, "completed", 3.0, 2, log=True))
        self.summary["round_5"][0]["step"] = 20
        with self.assertRaisesRegex(post_analysis.AnalysisError, "expected 50"):
            post_analysis.analyze(self.summary, self.sweep_dir)
        self.summary["round_5"][0]["step"] = 50
        self.summary["ranking"][0]["mean_val_loss"] = 9
        with self.assertRaisesRegex(post_analysis.AnalysisError, "summary.ranking disagrees"):
            post_analysis.analyze(self.summary, self.sweep_dir)

    def test_unknown_transform_is_not_invented_and_extra_rounds_are_rejected(self):
        self.summary["config"].pop("transform")
        data = post_analysis.analyze(self.summary, self.sweep_dir)
        setup = "".join(post_analysis._notebook_setup(data))
        self.assertIn("does not establish a specific alpha transform", setup)
        self.assertNotIn("$\\epsilon=", setup)
        self.summary["round_grids"]["round_4"] = [0.5]
        with self.assertRaisesRegex(post_analysis.AnalysisError, "Unexpected round keys"):
            post_analysis.analyze(self.summary, self.sweep_dir)
        del self.summary["round_grids"]["round_4"]
        for rounds in (0, -1, True, 3.5):
            self.summary["config"]["rounds"] = rounds
            with self.assertRaisesRegex(post_analysis.AnalysisError, "config.rounds"):
                post_analysis.analyze(self.summary, self.sweep_dir)

    def _check_short_sweep(self, count):
        final_stage = self._short_sweep(count)
        row = self.summary[final_stage].pop(1)
        with self.assertRaisesRegex(post_analysis.AnalysisError, "missing required {} result".format(final_stage)):
            post_analysis.analyze(self.summary, self.sweep_dir)
        self.summary[final_stage].insert(1, row)
        row["step"] = 10
        with self.assertRaisesRegex(post_analysis.AnalysisError, "expected 500"):
            post_analysis.analyze(self.summary, self.sweep_dir)
        row["step"] = 500
        self.summary["ranking"][0]["mean_val_loss"] = 9
        with self.assertRaisesRegex(post_analysis.AnalysisError, "summary.ranking disagrees"):
            post_analysis.analyze(self.summary, self.sweep_dir)
        self.summary["ranking"][0]["mean_val_loss"] = 3.1
        self._write_summary()
        data, destination = post_analysis.create_analysis(self.sweep_dir)
        self.assertEqual(len(data["stage_order"]), count)
        self.assertEqual(data["winners"][final_stage]["expected_seeds"], [0, 1])
        self.assertEqual(len(data["winners"][final_stage]["curves"]), 2)
        self.assertEqual(data["best"]["alpha"], 0.5)
        report = (destination / "experiment_report.md").read_text().strip()
        self.assertEqual(len(report.split("\n\n")), 2)
        self.assertIn("round {}: 500 steps".format(count), report)
        if count == 1:
            self.assertNotIn("Earlier rounds", report)
        else:
            self.assertIn("round 1: 10 steps", report)
        notebook = json.loads((destination / "analysis.ipynb").read_text())
        setup = "".join(notebook["cells"][0]["source"])
        self.assertIn("Round {}: 500 steps".format(count), setup)
        if count == 1:
            self.assertNotIn("Earlier rounds", setup)
        cell = notebook["cells"][2]
        self.assertEqual(len(cell["outputs"]), 5)
        import matplotlib.pyplot as plt
        shown = []
        original_show = plt.show
        try:
            plt.show = lambda *args, **kwargs: shown.append(plt.gcf())
            exec("".join(cell["source"]), {})
        finally:
            plt.show = original_show
            plt.close("all")
        self.assertEqual(len(shown), 5)
        self.assertEqual([ax.get_title() for ax in shown[0].axes],
                         ["Round {}".format(i) for i in range(1, count + 1)])
        self.assertEqual(len(shown[1].axes), count)
        self.assertEqual(shown[3].axes[0].get_title(), "Round {} seed variability".format(count))

    def test_one_round_uses_finalist_budget_and_seeds(self):
        self._check_short_sweep(1)

    def test_two_rounds_use_screen_then_finalist(self):
        self._check_short_sweep(2)

    def test_manifest_rerun_is_idempotent_and_different_summary_is_refused(self):
        _, destination = post_analysis.create_analysis(self.sweep_dir)
        prior_manifest = (destination / "analysis_manifest.json").read_bytes()
        _, same_destination = post_analysis.create_analysis(self.sweep_dir)
        self.assertEqual(same_destination, destination)
        self.assertEqual((destination / "analysis_manifest.json").read_bytes(), prior_manifest)
        self.summary["config"]["val_tokens"] += 1
        self._write_summary()
        with self.assertRaisesRegex(post_analysis.AnalysisError, "different summary.json"):
            post_analysis.create_analysis(self.sweep_dir)
        self.assertEqual((destination / "analysis_manifest.json").read_bytes(), prior_manifest)

    def test_missing_winner_log_fails_without_publishing_notebook(self):
        log_path = self.sweep_dir / "screening-0.0-0" / "console.log"
        log_path.unlink()
        destination = self.root / "no-winner-log"
        with self.assertRaisesRegex(post_analysis.AnalysisError, "missing console.log"):
            post_analysis.create_analysis(self.sweep_dir, destination)
        self.assertFalse(destination.exists())

    def test_missing_required_result_fails_without_publishing_notebook(self):
        self.summary["confirmation"] = [
            row for row in self.summary["confirmation"]
            if not (row["alpha"] == 1.0 and row["seed"] == 0)
        ]
        self._write_summary()
        destination = self.root / "incomplete-summary"
        with self.assertRaisesRegex(post_analysis.AnalysisError, "missing required confirmation result"):
            post_analysis.create_analysis(self.sweep_dir, destination)
        self.assertFalse(destination.exists())

    def test_completed_step_and_winner_curve_must_match_summary(self):
        screen_winner = self.summary["screening"][0]
        screen_winner["step"] -= 1
        self._write_summary()
        with self.assertRaisesRegex(post_analysis.AnalysisError, "expected 10"):
            post_analysis.create_analysis(self.sweep_dir, self.root / "wrong-step")

        screen_winner["step"] += 1
        log_path = Path(screen_winner["directory"]) / "console.log"
        log_path.write_text("step:5/10 val_loss:6.0000\nstep:10/10 val_loss:6.0000\n")
        self._write_summary()
        with self.assertRaisesRegex(post_analysis.AnalysisError, "final console.log loss"):
            post_analysis.create_analysis(self.sweep_dir, self.root / "wrong-final-loss")

    def test_summary_best_and_ranking_disagreements_are_rejected(self):
        self.summary["ranking"][0]["mean_val_loss"] += 0.1
        self._write_summary()
        with self.assertRaisesRegex(post_analysis.AnalysisError, "summary.ranking disagrees"):
            post_analysis.create_analysis(self.sweep_dir, self.root / "bad-ranking")
        self.summary["ranking"][0]["mean_val_loss"] -= 0.1
        self.summary["best_alpha"] = 1.0
        self._write_summary()
        with self.assertRaisesRegex(post_analysis.AnalysisError, "summary.best_alpha disagrees"):
            post_analysis.create_analysis(self.sweep_dir, self.root / "bad-best")

    def test_cli_flags_and_module_import_are_safe(self):
        prior_argv = sys.argv
        try:
            sys.argv = ["post_analysis.py", "--not-a-real-argument"]
            output = io.StringIO()
            with redirect_stdout(output):
                reloaded = importlib.reload(post_analysis)
            self.assertEqual(output.getvalue(), "")
            self.assertTrue(callable(reloaded.main))
            destination = self.root / "cli-output"
            self.assertEqual(
                reloaded.main(["--sweep-dir", str(self.sweep_dir),
                               "--output-dir", str(destination)]),
                0,
            )
            self.assertTrue((destination / "analysis_manifest.json").is_file())
        finally:
            sys.argv = prior_argv

    def test_running_sweep_is_refused(self):
        self.summary["status"] = "running"
        self._write_summary()
        with self.assertRaisesRegex(post_analysis.AnalysisError, "only completed runs"):
            post_analysis.create_analysis(self.sweep_dir)


if __name__ == "__main__":
    unittest.main()
