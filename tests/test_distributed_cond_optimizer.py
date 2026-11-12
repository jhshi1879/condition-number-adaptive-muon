import ast
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


TRAINER = Path(__file__).resolve().parent.parent / "train_gpt2_cond_muon.py"
source = ast.parse(TRAINER.read_text())
definitions = [
    node for node in source.body
    if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    and node.name in {"polynomial_fractional_power", "moment_fractional_power", "ConditionalNumberAdaptiveMuon"}
]
namespace = {"torch": torch, "dist": dist, "math": math}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(TRAINER), "exec"), namespace)
Optimizer = namespace["ConditionalNumberAdaptiveMuon"]


def distributed_worker(rank, world_size, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size)
    try:
        shapes = [(6, 2), (4, 2), (2, 2), (3, 2)]
        params = [torch.nn.Parameter(torch.arange(math.prod(s), dtype=torch.float32).reshape(s) / 10) for s in shapes]
        reference = [torch.nn.Parameter(p.detach().clone()) for p in params]
        optimizer = Optimizer(params, lr=0.01)
        baseline = Optimizer(reference, lr=0.01)
        actual_transform = namespace["moment_fractional_power"]

        for step in range(4):
            for i, (p, q) in enumerate(zip(params, reference)):
                grad = None if i == 3 and step == 0 else (
                    torch.arange(p.numel(), dtype=torch.float32).reshape_as(p) + 1 + step
                ) / (i + 2)
                p.grad = None if grad is None else grad.clone()
                q.grad = None if grad is None else grad.clone()

            with mock.patch.dict(namespace, moment_fractional_power=mock.Mock(wraps=actual_transform)):
                optimizer.step()
                local_calls = namespace["moment_fractional_power"].call_count
            calls = torch.tensor(local_calls)
            dist.all_reduce(calls)
            expected_calls = sum(
                (3 if p.size(0) == 3 * p.size(1) else 1)
                for p in params if p.grad is not None
            )
            assert calls.item() == expected_calls
            assert local_calls < expected_calls

            with mock.patch.object(dist, "is_initialized", return_value=False):
                baseline.step()
            for p, q in zip(params, reference):
                torch.testing.assert_allclose(p, q, rtol=1e-5, atol=1e-6)
                if q.grad is None:
                    continue
                actual_state = optimizer.state[p]
                expected_state = baseline.state[q]
                assert actual_state["step"] == expected_state["step"]
                torch.testing.assert_allclose(actual_state["momentum_buffer"], expected_state["momentum_buffer"])
                for alpha, expected in zip(actual_state["alphas"], expected_state["alphas"]):
                    torch.testing.assert_allclose(alpha, expected)

            if step == 1:
                saved = [optimizer.state_dict() if rank == 0 else None]
                dist.broadcast_object_list(saved, src=0)
                optimizer = Optimizer(params, lr=0.01)
                optimizer.load_state_dict(saved[0])
    finally:
        dist.destroy_process_group()


class DistributedConditionalMuonTests(unittest.TestCase):
    def test_moment_estimate_and_ema(self):
        gradient = torch.diag(torch.tensor([1.0, 0.05, 0.01, 0.001]))
        parameter = torch.nn.Parameter(torch.zeros_like(gradient))
        optimizer = Optimizer([parameter], lr=0.01, momentum=0.5, kappa0=1.1)
        parameter.grad = gradient.clone()
        with mock.patch.object(torch.linalg, "svd", side_effect=AssertionError("SVD must not be used")):
            optimizer.step()
        state = optimizer.state[parameter]
        torch.testing.assert_allclose(state["momentum_buffer"], 0.5 * gradient)
        x = gradient / gradient.norm()
        gram = x @ x.T
        m2 = gram.diagonal().sum() / 4
        m4 = gram.square().sum() / 4
        log_kappa = 1.64485 * torch.sqrt(torch.log(m4 / m2.square()))
        expected_alpha = (1 - math.log(1.1) / log_kappa).clamp(0, 1)
        torch.testing.assert_allclose(state["alphas"][0], expected_alpha)
        self.assertAlmostEqual(parameter.norm().item(), 0.02, places=5)
        parameter.grad = -gradient.clone()
        optimizer.step()
        torch.testing.assert_allclose(state["momentum_buffer"], -0.25 * gradient)
        self.assertEqual(state["step"], 2)

    def test_zero_matrix_and_equal_spectrum(self):
        transform = namespace["moment_fractional_power"]
        for matrix in (torch.zeros(3, 2), torch.eye(2)):
            result, alpha = transform(matrix, 10.0, 1e-7)
            self.assertTrue(torch.isfinite(result).all())
            self.assertEqual(alpha.item(), 0.0)
        result, _ = transform(torch.zeros(3, 2), 10.0, 1e-7)
        self.assertEqual(result.abs().sum().item(), 0.0)

    def test_polynomial_error_and_regularization(self):
        transform = namespace["polynomial_fractional_power"]
        singular_values = torch.linspace(0.01, 1.0, 10001, dtype=torch.float64)
        nodes = torch.arange(33, dtype=torch.float64)
        theta = math.pi * (nodes + 0.5) / 33
        for alpha, max_error in [(0, 0.001), (0.25, 0.006), (0.5, 0.034),
                                 (0.75, 0.146), (1, 0.565)]:
            exponent = torch.tensor(1 - alpha, dtype=torch.float64)
            samples = ((theta.cos() + 1) * 0.5 + 0.01**2).pow((exponent - 1) / 2)
            coefficients = (2.0 / 33) * (torch.cos(nodes[:, None] * theta[None, :]) @ samples)
            coefficients[0] *= 0.5
            argument = 2 * singular_values.square() - 1
            b1 = torch.zeros_like(argument)
            b2 = torch.zeros_like(argument)
            for k in range(32, 0, -1):
                b1, b2 = 2 * argument * b1 - b2 + coefficients[k], b1
            actual = singular_values * (argument * b1 - b2 + coefficients[0])
            error = (actual - singular_values.pow(exponent)).abs().max().item()
            self.assertLess(error, max_error, f"alpha={alpha}: error={error}")
            x = torch.diag(torch.tensor([0.0, 0.005, 0.01, 0.5, 1.0]))
            actual_matrix = transform(x, x @ x.T, exponent.float())
            self.assertTrue(torch.isfinite(actual_matrix).all())
            self.assertEqual(actual_matrix[0, 0].item(), 0.0)
            for index in (2, 3, 4):
                sigma = x[index, index].item()
                self.assertLess(abs(actual_matrix[index, index].item() - sigma**exponent.item()), max_error + 1e-4)

    def test_two_rank_updates_match_single_rank_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / "rendezvous")
            mp.spawn(distributed_worker, args=(2, rendezvous), nprocs=2, join=True)

    def test_three_rank_updates_match_single_rank_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / "rendezvous")
            mp.spawn(distributed_worker, args=(3, rendezvous), nprocs=3, join=True)


if __name__ == "__main__":
    unittest.main()
