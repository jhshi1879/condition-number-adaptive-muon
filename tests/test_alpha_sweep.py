import ast
import argparse
import contextlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import unittest
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "alpha_sweep"
TRAINERS = ("train_gpt2.py", "train_gpt2_adamw.py", "train_gpt2_cond_muon.py")
sys.path.insert(0, str(SWEEP))
from models import GPT, GPTConfig


def sweep_namespace():
    trainer = SWEEP / "train_gpt2_cond_muon.py"
    names = {
        "polynomial_fractional_power", "moment_fractional_power",
        "ConditionalNumberAdaptiveMuon", "write_json", "refine_alpha_grid", "run_alpha_sweep",
    }
    definitions = [
        node for node in ast.parse(trainer.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    namespace = dict(
        torch=torch, dist=dist, math=math, json=json, os=os, sys=sys,
        Path=Path, statistics=statistics, subprocess=subprocess, uuid=uuid,
        __file__=str(trainer),
    )
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(trainer), "exec"), namespace)
    return namespace


class AlphaSweepTests(unittest.TestCase):
    def test_fixed_alpha_matches_regularized_svd_through_three(self):
        transform = sweep_namespace()["moment_fractional_power"]
        matrix = torch.tensor([[1.0, 0.2, 0.0], [0.1, 0.15, 0.0]])
        for value in (matrix, matrix.T, torch.zeros_like(matrix)):
            x = value / value.norm().clamp_min(1e-7)
            u, s, vh = torch.linalg.svd(x, full_matrices=False)
            for alpha in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0):
                with self.subTest(shape=value.shape, alpha=alpha):
                    actual, recorded = transform(value, 10.0, 1e-7, alpha=alpha)
                    expected = (u * (s * (s.square() + 0.01**2).pow(-alpha / 2))) @ vh
                    torch.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-5)
                    self.assertEqual(recorded.item(), alpha)
                    self.assertTrue(torch.isfinite(actual).all())

    def test_fixed_alpha_optimizer_accepts_above_one_and_rejects_invalid_values(self):
        optimizer_class = sweep_namespace()["ConditionalNumberAdaptiveMuon"]
        parameter = torch.nn.Parameter(torch.zeros(6, 2))
        optimizer = optimizer_class([parameter], alpha=3.0, lr=0.01)
        parameter.grad = torch.tensor([[1.0, 0.0], [0.0, 0.2]]).repeat(3, 1)
        optimizer.step()
        self.assertTrue(torch.isfinite(parameter).all())
        self.assertAlmostEqual(parameter.norm().item(), math.sqrt(6) * 0.01, places=5)
        self.assertEqual([a.item() for a in optimizer.state[parameter]["alphas"]], [3.0] * 3)
        for invalid in (-0.25, float("nan"), float("inf")):
            with self.subTest(alpha=invalid), self.assertRaises(ValueError):
                optimizer_class([parameter], alpha=invalid)

    def test_refined_grids_merge_overlaps_and_stay_inside_search_bounds(self):
        refine = sweep_namespace()["refine_alpha_grid"]
        actual = refine([0.0, 0.25], 0.25, 4, 0.0, 3.0)
        self.assertEqual(actual, [index / 16 for index in range(9)])
        self.assertEqual(refine([3.0], 0.0625, 2, 0.0, 3.0), [2.9375, 2.96875, 3.0])

    def test_three_round_sweep_refines_and_ranks_seed_means(self):
        namespace = sweep_namespace()
        with tempfile.TemporaryDirectory() as directory:
            options = SimpleNamespace(
                output_dir=Path(directory) / "sweep",
                alpha_grid=[0.0, 0.25, 1.5, 3.0],
                screen_steps=100, refine_steps=100, finalist_steps=500, seed=7,
                finalist_seeds=[7, 8], top_k=2, device_batch_size=8,
                val_tokens=1048576, val_loss_every=25, save_every=0,
                warmdown_iters=10,
            )
            commands = []

            def child(command, **kwargs):
                commands.append(command)
                self.assertIn("--no-checkpoints", command)
                alpha = float(command[command.index("--alpha") + 1])
                seed = int(command[command.index("--seed") + 1])
                steps = int(command[command.index("--num-iterations") + 1])
                result_file = Path(command[command.index("--result-file") + 1])
                stage = result_file.parent.name.split("-")[0]
                self.assertEqual(kwargs["env"]["STOP_RUNPOD_ON_EXIT"], "0")
                if alpha == 3:
                    status, loss, returncode = "diverged", None, 1
                elif stage == "screening":
                    status, loss, returncode = "completed", {0: 6.0, 0.25: 5.0, 1.5: 4.0}[alpha], 0
                elif stage == "refinement":
                    status, loss, returncode = "completed", 4 + (alpha - 1.5)**2, 0
                else:
                    status, loss, returncode = "completed", (
                        {7: 3.0, 8: 5.0}[seed] if alpha == 1.5 else
                        {7: 3.4, 8: 3.6}[seed] + (alpha - 1.46875)**2
                    ), 0
                namespace["write_json"](result_file, dict(
                    status=status, alpha=alpha, seed=seed, step=steps, final_val_loss=loss,
                ))
                return SimpleNamespace(returncode=returncode)

            with mock.patch.object(subprocess, "run", side_effect=child), mock.patch("builtins.print"):
                result = namespace["run_alpha_sweep"](options)
            self.assertEqual(len(commands), 36)
            self.assertEqual(result["status"], "completed_with_divergences")
            self.assertEqual(list(result), [
                "status", "config", "screening", "refinement", "confirmation",
                "round_grids", "ranking", "best_alpha", "best_at_grid_boundary",
            ])
            self.assertNotIn("rounds", result["config"])
            self.assertEqual(result["best_alpha"], 1.46875)
            self.assertEqual(len(result["refinement"]), 18)
            self.assertEqual(len(result["confirmation"]), 14)
            self.assertIn(0.0, result["round_grids"]["refinement"])
            self.assertIn(1.75, result["round_grids"]["refinement"])
            self.assertEqual(result["round_grids"]["confirmation"], [
                1.375, 1.40625, 1.4375, 1.46875, 1.5, 1.53125, 1.5625,
            ])
            self.assertTrue(all(row["seed"] == 7 for row in result["screening"] + result["refinement"]))
            self.assertEqual({row["seed"] for row in result["confirmation"]}, {7, 8})
            self.assertEqual(result["ranking"][0]["seed_losses"], [3.4, 3.6])
            self.assertEqual(result["ranking"][0]["mean_val_loss"], 3.5)
            self.assertNotIn("active_trial", result)
            saved = json.loads((options.output_dir / "summary.json").read_text())
            self.assertEqual(saved, result)
            with self.assertRaises(FileExistsError):
                namespace["run_alpha_sweep"](options)

    def test_five_round_sweep_refines_each_round_and_confirms_with_two_seeds(self):
        namespace = sweep_namespace()
        with tempfile.TemporaryDirectory() as directory:
            options = SimpleNamespace(
                output_dir=Path(directory) / "sweep",
                alpha_grid=[index / 4 for index in range(41)], rounds=5,
                screen_steps=100, refine_steps=100, finalist_steps=500, seed=0,
                finalist_seeds=[0, 1], top_k=2, device_batch_size=8,
                val_tokens=1048576, val_loss_every=25, save_every=0,
                warmdown_iters=10,
            )
            commands = []

            def child(command, **kwargs):
                commands.append(command)
                self.assertIn("--no-checkpoints", command)
                alpha = float(command[command.index("--alpha") + 1])
                seed = int(command[command.index("--seed") + 1])
                steps = int(command[command.index("--num-iterations") + 1])
                result_file = Path(command[command.index("--result-file") + 1])
                diverged = alpha == 10
                namespace["write_json"](result_file, dict(
                    status="diverged" if diverged else "completed",
                    alpha=alpha, seed=seed, step=steps,
                    final_val_loss=None if diverged else (alpha - 3.125)**2 + seed,
                ))
                return SimpleNamespace(returncode=int(diverged))

            with mock.patch.object(subprocess, "run", side_effect=child), mock.patch("builtins.print"):
                result = namespace["run_alpha_sweep"](options)

            stages = [f"round_{index}" for index in range(1, 6)]
            self.assertEqual(result["config"]["rounds"], 5)
            self.assertEqual(set(result["round_grids"]), set(stages))
            self.assertFalse({"screening", "refinement", "confirmation"} & result.keys())
            self.assertEqual(result["round_grids"]["round_1"], options.alpha_grid)
            radius = 0.25
            for index in range(1, 5):
                preceding = result[stages[index - 1]]
                centers = [row["alpha"] for row in sorted(
                    (row for row in preceding if row["status"] == "completed"),
                    key=lambda row: (row["final_val_loss"], row["alpha"]),
                )[:2]]
                subdivisions = 4 if index == 1 else 2
                expected = sorted({
                    round(center + radius * offset / subdivisions, 12)
                    for center in centers for offset in range(-subdivisions, subdivisions + 1)
                    if 0 <= round(center + radius * offset / subdivisions, 12) <= 10
                })
                self.assertEqual(result["round_grids"][stages[index]], expected)
                radius /= subdivisions
            for stage in stages[:-1]:
                self.assertTrue(all(row["seed"] == 0 and row["step"] == 100 for row in result[stage]))
            self.assertEqual({row["seed"] for row in result[stages[-1]]}, {0, 1})
            self.assertTrue(all(row["step"] == 500 for row in result[stages[-1]]))
            self.assertEqual(result["best_alpha"], 3.125)
            self.assertEqual(result["ranking"][0]["seed_losses"], [0.0, 1.0])
            self.assertEqual(result["status"], "completed_with_divergences")
            self.assertEqual(json.loads((options.output_dir / "summary.json").read_text()), result)

    def test_checkpoint_condition_for_direct_training(self):
        source = ast.parse((SWEEP / "train_gpt2_cond_muon.py").read_text())
        loop = next(node for node in source.body if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == "step")
        checkpoint = next(
            node for node in loop.body if isinstance(node, ast.If)
            and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "torch" and child.func.attr == "save"
                    for child in ast.walk(node))
        )
        code = compile(ast.Module(body=[checkpoint], type_ignores=[]), "<checkpoint>", "exec")
        for disabled, interval, step, last, expected in (
            (False, 0, 10, True, True),
            (False, 0, 0, False, False),
            (False, 5, 5, False, True),
            (True, 0, 10, True, False),
            (True, 5, 5, False, False),
        ):
            with self.subTest(disabled=disabled, interval=interval, step=step, last=last):
                fake_torch = mock.Mock()
                env = dict(
                    master_process=True, options=SimpleNamespace(no_checkpoints=disabled, alpha=1, seed=0),
                    args=SimpleNamespace(save_every=interval), step=step, last_step=last,
                    torch=fake_torch, time=SimpleNamespace(time=lambda: 1), t0=0,
                    training_time_ms=0, code="source", raw_model=mock.Mock(),
                    optimizers=[mock.Mock()], os=os, logdir="logs",
                )
                exec(code, env)
                self.assertEqual(fake_torch.save.call_count, int(expected))
                if expected:
                    self.assertEqual(fake_torch.save.call_args.args[1], f"logs/state_step{step:06d}.pt")

    def test_one_and_two_round_sweeps(self):
        namespace = sweep_namespace()
        for rounds, initial in ((1, [3.0]), (2, [3.0, 3.25, 3.5])):
            with self.subTest(rounds=rounds), tempfile.TemporaryDirectory() as directory:
                options = SimpleNamespace(
                    output_dir=Path(directory) / "sweep", alpha_grid=initial, rounds=rounds,
                    screen_steps=100, refine_steps=100, finalist_steps=500, seed=7,
                    finalist_seeds=[0, 1], top_k=2, device_batch_size=8,
                    val_tokens=1048576, val_loss_every=25, save_every=0,
                    warmdown_iters=10,
                )
                commands = []

                def child(command, **kwargs):
                    commands.append(command)
                    alpha = float(command[command.index("--alpha") + 1])
                    seed = int(command[command.index("--seed") + 1])
                    steps = int(command[command.index("--num-iterations") + 1])
                    result_path = Path(command[command.index("--result-file") + 1])
                    namespace["write_json"](result_path, dict(
                        status="completed", alpha=alpha, seed=seed, step=steps,
                        final_val_loss=(alpha - 3.25)**2 + seed,
                    ))
                    return SimpleNamespace(returncode=0)

                with mock.patch.object(subprocess, "run", side_effect=child), mock.patch("builtins.print"):
                    result = namespace["run_alpha_sweep"](options)
                stages = [f"round_{index}" for index in range(1, rounds + 1)]
                self.assertEqual(result["config"]["rounds"], rounds)
                self.assertEqual(set(result["round_grids"]), set(stages))
                self.assertFalse({"screening", "refinement", "confirmation"} & result.keys())
                self.assertTrue(all("--no-checkpoints" in command for command in commands))
                self.assertEqual(result["round_grids"]["round_1"], initial)
                if rounds == 2:
                    self.assertTrue(all(row["seed"] == 7 and row["step"] == 100
                                        for row in result["round_1"]))
                    self.assertEqual(result["round_grids"]["round_2"],
                                     [3 + index / 16 for index in range(9)])
                final = result[stages[-1]]
                self.assertEqual({row["seed"] for row in final}, {0, 1})
                self.assertTrue(all(row["step"] == 500 for row in final))
                self.assertEqual(len(final), 2 * len(result["round_grids"][stages[-1]]))
                self.assertEqual(result["best_alpha"], 3.0 if rounds == 1 else 3.25)
                self.assertEqual(json.loads((options.output_dir / "summary.json").read_text()), result)

    def test_sweep_aborts_unexpected_child_failure_with_persistent_status(self):
        namespace = sweep_namespace()
        with tempfile.TemporaryDirectory() as directory:
            options = SimpleNamespace(
                output_dir=Path(directory) / "sweep", alpha_grid=[0.0, 3.0],
                screen_steps=100, refine_steps=100, finalist_steps=500, seed=0,
                finalist_seeds=[0, 1], top_k=2, device_batch_size=8,
                val_tokens=1048576, val_loss_every=25, save_every=0,
                warmdown_iters=10,
            )
            with mock.patch.object(subprocess, "run", return_value=SimpleNamespace(returncode=1)), \
                 mock.patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "without a result"):
                namespace["run_alpha_sweep"](options)
            saved = json.loads((options.output_dir / "summary.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertIsNone(saved["best_alpha"])
            self.assertEqual(saved["active_trial"]["alpha"], 0.0)

    def test_default_model_size_and_tied_embeddings(self):
        config = GPTConfig()
        model = GPT(config)
        self.assertEqual((config.n_layer, config.n_head, config.n_embd), (6, 6, 384))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 29_915_520)
        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)

    @unittest.skipUnless(hasattr(F, "scaled_dot_product_attention"), "requires PyTorch 2")
    def test_model_forward_and_backward(self):
        model = GPT(GPTConfig(vocab_size=32))
        tokens = torch.randint(0, 32, (2, 4))
        logits, loss = model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 4, 32))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.transformer.h[0].attn.c_attn.weight.grad)

    def test_trainers_use_repository_data(self):
        for name in TRAINERS:
            trainer = SWEEP / name
            with self.subTest(trainer=trainer.name):
                source = ast.parse(trainer.read_text())
                definition = next(
                    node for node in source.body
                    if isinstance(node, ast.ClassDef) and node.name == "Hyperparameters"
                )
                namespace = {"dataclass": dataclass, "os": os, "__file__": str(trainer)}
                exec(compile(ast.Module(body=[definition], type_ignores=[]), str(trainer), "exec"), namespace)
                args = namespace["Hyperparameters"]()
                self.assertEqual(Path(args.input_bin).resolve(), ROOT / "data/fineweb10B/fineweb_train_*.bin")
                self.assertEqual(Path(args.input_val_bin).resolve(), ROOT / "data/fineweb10B/fineweb_val_*.bin")

    @unittest.skipUnless(hasattr(torch, "compile"), "requires PyTorch 2")
    def test_trainers_help_from_both_directories(self):
        for trainer in TRAINERS:
            for cwd in (ROOT, SWEEP):
                with self.subTest(trainer=trainer, cwd=cwd):
                    result = subprocess.run(
                        [sys.executable, str(SWEEP / trainer), "--help"],
                        cwd=cwd, text=True, capture_output=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--num-iterations", result.stdout)
                    if trainer == "train_gpt2_cond_muon.py":
                        self.assertIn("--rounds", result.stdout)
                        self.assertIn("--no-checkpoints", result.stdout)

    def test_checkpoint_and_rounds_cli_options(self):
        trainer = SWEEP / "train_gpt2_cond_muon.py"
        source = ast.parse(trainer.read_text())
        definition = next(node for node in source.body if isinstance(node, ast.ClassDef)
                          and node.name == "Hyperparameters")
        start = next(index for index, node in enumerate(source.body)
                     if isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == "defaults"
                         for target in node.targets))
        end = next(index for index, node in enumerate(source.body[start:], start)
                   if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute)
                   and isinstance(node.test.value, ast.Name) and node.test.value.id == "options"
                   and node.test.attr == "sweep")
        code = compile(ast.Module(body=[definition] + source.body[start:end + 1], type_ignores=[]),
                       str(trainer), "exec")
        for arguments, disabled in (([], False), (["--no-checkpoints"], True)):
            env = dict(dataclass=dataclass, os=os, __file__=str(trainer), argparse=argparse,
                       Path=Path, math=math)
            with mock.patch.object(sys, "argv", ["trainer"] + arguments):
                exec(code, env)
            self.assertEqual(env["options"].rounds, 3)
            self.assertEqual(env["options"].no_checkpoints, disabled)
        env = dict(dataclass=dataclass, os=os, __file__=str(trainer), argparse=argparse,
                   Path=Path, math=math)
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", ["trainer", "--sweep", "--rounds", "0"]), \
             contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            exec(code, env)
        self.assertIn("at least one round", stderr.getvalue())
        run = mock.Mock()
        env = dict(dataclass=dataclass, os=os, __file__=str(trainer), argparse=argparse,
                   Path=Path, math=math, run_alpha_sweep=run,
                   sys=SimpleNamespace(exit=mock.Mock(side_effect=SystemExit)))
        with mock.patch.object(sys, "argv", ["trainer", "--sweep", "--rounds", "1",
                                            "--alpha-grid", "3"]), self.assertRaises(SystemExit):
            exec(code, env)
        self.assertEqual(run.call_args.args[0].rounds, 1)


if __name__ == "__main__":
    unittest.main()
