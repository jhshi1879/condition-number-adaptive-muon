import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


LAUNCHER = Path(__file__).resolve().parent.parent / "launch_tmux.sh"
DOWNLOADER = LAUNCHER.parent / "data" / "cached_fineweb10B.py"

FAKE_PYTHON = """\
#!/bin/bash
set -eu
echo "$*" >> "$CALLS"
if [[ ${CUDA_FAIL:-0} == 1 && "$*" == *"import torch"* ]]; then
    echo "CUDA check failed" >&2
    exit 1
fi
if [[ $1 == -m && $2 == venv ]]; then
    mkdir -p "$4/bin"
    cp "$FAKE_PYTHON" "$4/bin/python"
elif [[ $1 == -c && $2 == *find_spec* ]]; then
    if [[ ! -f "$INSTALLED" ]]; then
        if [[ ${3:-0} != 0 ]]; then
            echo "numpy runpod huggingface_hub"
        else
            echo "numpy runpod"
        fi
    fi
elif [[ $1 == -m && $2 == pip ]]; then
    if [[ ${PIP_FAIL:-0} == 1 ]]; then
        echo "pip failed" >&2
        exit 1
    fi
    touch "$INSTALLED"
fi
"""


class PodSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.venv = root / "venv"
        self.calls = root / "calls"
        bin_dir = root / "bin"
        bin_dir.mkdir()
        fake_python = bin_dir / "python3"
        fake_python.write_text(FAKE_PYTHON)
        fake_python.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_PYTHON": str(fake_python),
            "CALLS": str(self.calls),
            "INSTALLED": str(root / "installed"),
        }

    def run_setup(self, shards=0):
        return subprocess.run(
            ["bash", str(LAUNCHER), "--pod-setup", str(self.venv), str(shards)],
            env=self.env, text=True, capture_output=True,
        )

    def test_create_and_reuse_environment(self):
        first = self.run_setup()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("-m venv --system-site-packages", self.calls.read_text())
        self.assertIn("-m pip install numpy runpod>=2.0,<3", self.calls.read_text())
        self.calls.write_text("")
        second = self.run_setup()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertNotIn("-m venv", self.calls.read_text())
        self.assertNotIn("-m pip install", self.calls.read_text())

    def test_broken_existing_environment_is_not_overwritten(self):
        self.venv.mkdir()
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inspect it before retrying", result.stderr)
        self.assertFalse(self.calls.exists())

    def test_cuda_failure_prevents_environment_creation(self):
        self.env["CUDA_FAIL"] = "1"
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA check failed", result.stderr)
        self.assertFalse(self.venv.exists())

    def test_install_failure_is_reported(self):
        self.env["PIP_FAIL"] = "1"
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pip failed", result.stderr)
        self.assertNotIn("prerequisites are ready", result.stdout)

    def test_downloader_dependency_is_installed_when_requested(self):
        result = self.run_setup(51)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-m pip install numpy runpod>=2.0,<3 huggingface_hub", self.calls.read_text())
        self.assertIn("-c import huggingface_hub", self.calls.read_text())


class DownloadTests(unittest.TestCase):
    def test_existing_shards_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            shards = data / "fineweb10B"
            shards.mkdir()
            downloader_dir = root / "downloader"
            downloader_dir.mkdir()
            (downloader_dir / "fineweb10B").symlink_to(shards)
            downloader = downloader_dir / DOWNLOADER.name
            downloader.write_text(DOWNLOADER.read_text())
            existing_train = shards / "fineweb_train_000001.bin"
            existing_val = shards / "fineweb_val_000000.bin"
            existing_train.write_bytes(b"existing train")
            existing_val.write_bytes(b"existing validation")
            (root / "huggingface_hub.py").write_text(
                "from pathlib import Path\n"
                "def hf_hub_download(repo_id, filename, repo_type, local_dir):\n"
                "    Path(local_dir, filename).write_bytes(b'downloaded')\n"
            )
            result = subprocess.run(
                [sys.executable, str(downloader), "2"],
                env={**os.environ, "PYTHONPATH": str(root)}, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(existing_train.read_bytes(), b"existing train")
            self.assertEqual(existing_val.read_bytes(), b"existing validation")
            self.assertEqual((shards / "fineweb_train_000002.bin").read_bytes(), b"downloaded")

    def test_invalid_download_counts_fail_before_connecting(self):
        for value in ("0", "104", "-1", "abc", "01", ""):
            with self.subTest(value=value):
                result = subprocess.run(
                    ["bash", str(LAUNCHER), "127.0.0.1", "22", "pod", f"--download_shards={value}"],
                    text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("integer from 1 to 103", result.stderr)

    def test_flag_uploads_downloader_and_is_forwarded_to_session(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".ssh").mkdir()
            (root / ".ssh" / "id_ed25519").touch()
            bin_dir = root / "bin"
            bin_dir.mkdir()
            calls = root / "calls"
            for name in ("ssh", "scp"):
                executable = bin_dir / name
                executable.write_text('#!/bin/bash\nprintf "%s\\n" "$0 $*" >> "$CALLS"\n')
                executable.chmod(0o755)
            env = {
                **os.environ,
                "HOME": str(root),
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "CALLS": str(calls),
            }
            trainer = LAUNCHER.parent / "train_gpt2_adamw.py"
            result = subprocess.run(
                ["bash", str(LAUNCHER), "127.0.0.1", "22", "pod", str(trainer), "--download_shards=51"],
                env=env, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = calls.read_text()
            self.assertIn("--pod-setup /workspace/muon-venv 51", commands)
            self.assertIn(str(DOWNLOADER), commands)
            self.assertIn("downloader/cached_fineweb10B.py", commands)
            self.assertIn("/workspace/muon-data/data/fineweb10B", commands)
            self.assertIn(" pod 51", commands)
            self.assertIn("tmux new-session -s 'train_gpt2_adamw-", commands)
            self.assertIn("tmux attach -t train_gpt2_adamw-", result.stdout)

            calls.write_text("")
            result = subprocess.run(
                ["bash", str(LAUNCHER), "127.0.0.1", "22", "pod"],
                env=env, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("tmux new-session -s 'train_gpt2-", calls.read_text())

            unusual_trainer = root / "my model's.test.py"
            unusual_trainer.touch()
            calls.write_text("")
            result = subprocess.run(
                ["bash", str(LAUNCHER), "127.0.0.1", "22", "pod", str(unusual_trainer)],
                env=env, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("tmux new-session -s 'my_model_s_test-", calls.read_text())


if __name__ == "__main__":
    unittest.main()
