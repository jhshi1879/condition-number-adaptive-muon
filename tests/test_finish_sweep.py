from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("finish_sweep", ROOT / "alpha_sweep" / "finish_sweep.py")
finish = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finish)


class FinalizationTests(unittest.TestCase):
    def summary(self):
        config = dict(screening_seed=0, finalist_seeds=[0, 1])
        summary = dict(
            status="completed", config=config, best_alpha=1.5,
            ranking=[dict(alpha=1.5)],
            round_grids={stage: [0.0, 1.5] for stage in finish.STAGES},
        )
        for stage in finish.STAGES:
            seeds = [0, 1] if stage == "confirmation" else [0]
            summary[stage] = [
                dict(alpha=alpha, seed=seed, status="completed", final_val_loss=5.0)
                for alpha in (0.0, 1.5) for seed in seeds
            ]
        return summary

    def test_completion_requires_all_three_rounds_and_all_seeds(self):
        summary = self.summary()
        finish.validate_summary(summary, summary["config"])
        summary["confirmation"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            finish.validate_summary(summary, summary["config"])
        summary = self.summary()
        summary["status"] = "running"
        with self.assertRaisesRegex(ValueError, "completed"):
            finish.validate_summary(summary, summary["config"])

    def test_stop_credential_survives_ssh_environment_filtering(self):
        def ssh(config, command):
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(Path, "read_bytes", return_value=b"RUNPOD_API_KEY=private-test-token\0"), \
                 redirect_stdout(io.StringIO()) as output:
                exec(command[2], {})
            return output.getvalue()

        with mock.patch.object(finish, "ssh", side_effect=ssh), \
             mock.patch.object(finish, "runpod_request", return_value={"id": "test-pod"}):
            self.assertEqual(finish.read_token({"pod_id": "test-pod"}), "private-test-token")

    def test_environment_file_is_parsed_without_execution_or_credential_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('export RUNPOD_API_KEY="private-test-${TOKEN}" # local only\n')
            config = dict(pod_id="test-pod", env_file=str(path))
            with mock.patch.object(finish, "ssh") as ssh, \
                 mock.patch.object(finish, "runpod_request", return_value={"id": "test-pod"}):
                self.assertEqual(finish.read_token(config), "private-test-${TOKEN}")
                ssh.assert_not_called()
            path.write_text("OTHER_KEY=unused\n")
            with mock.patch.object(finish, "ssh") as ssh:
                with self.assertRaisesRegex(RuntimeError, "RUNPOD_API_KEY is missing"):
                    finish.read_token(config)
                ssh.assert_not_called()

    def test_skip_preflight_freezes_sources_without_api_or_git_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in finish.SOURCES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"pass\n")
            sweep = root / "sweep"
            sweep.mkdir()
            finish.atomic_json(sweep / "summary.json", dict(status="running", config={"seed": 0}))
            key = Path("/dev/shm/test-publisher/key")
            stat = Path.stat

            def key_stat(path, *args, **kwargs):
                return SimpleNamespace(st_mode=0o600) if path == key else stat(path, *args, **kwargs)

            options = SimpleNamespace(
                repo="owner/repo", branch="main", results_relative="alpha_sweep/results/test",
                ssh_key=key, pod_id="test-pod", project_root=root, sweep_dir=str(sweep),
                known_hosts=root / "hosts", deploy_key_id=42, state_dir=root / "state",
                skip_preflight=True,
            )
            with mock.patch.object(Path, "stat", key_stat), \
                 mock.patch.object(finish, "read_token", return_value="private-test-token") as token, \
                 mock.patch.object(finish, "git_command") as git, \
                 mock.patch.object(finish, "run") as run, mock.patch("builtins.print"):
                finish.prepare_pod(options)
                self.assertFalse(token.call_args.kwargs["verify"])
                git.assert_not_called()
                run.assert_not_called()
            self.assertEqual(json.loads((options.state_dir / "state.json").read_text())["phase"], "waiting")
            self.assertEqual((options.state_dir / "sources/alpha_sweep/launch.sh").read_bytes(), b"pass\n")

    def test_skipped_credential_probe_does_not_skip_runtime_authentication(self):
        with mock.patch.dict(os.environ, {"RUNPOD_API_KEY": "private-test-token"}), \
             mock.patch.object(finish, "runpod_request", return_value={"id": "test-pod"}) as request:
            config = dict(on_pod=True, pod_id="test-pod")
            self.assertEqual(finish.read_token(config, verify=False), "private-test-token")
            request.assert_not_called()
            finish.read_token(config)
            request.assert_called_once_with("private-test-token", "test-pod")

    def test_publication_rejects_checkpoints_and_credential_leaks(self):
        for files in (
            {"state.pt": b"checkpoint"},
            {"script.py": b"key = 'private-test-token'"},
            {"script.py": b"password = 'ghp_" + b"A" * 36 + b"'"},
        ):
            with self.subTest(paths=list(files)), self.assertRaises(ValueError):
                finish.validate_publish_files(files, "private-test-token")

    def test_publication_is_allowlisted_non_force_and_preserves_base_tree(self):
        files = {"alpha_sweep/post_analysis.py": b"print('analysis')\n"}
        digest = finish.blob_sha(next(iter(files.values())))
        calls = []
        updated = False

        def api(method, path, payload=None):
            nonlocal updated
            calls.append((method, path, payload))
            if "/ref/heads/" in path:
                return {"object": {"sha": "new-commit" if updated else "base-commit"}}
            if path.endswith("commits/base-commit"):
                return {"tree": {"sha": "base-tree"}}
            if path.endswith("commits/new-commit"):
                return {"tree": {"sha": "new-tree"}}
            if "trees/base-tree" in path:
                return {"tree": [{"path": "unrelated.txt", "type": "blob", "sha": "keep"}]}
            if path.endswith("/blobs"):
                return {"sha": digest}
            if path.endswith("/trees"):
                self.assertEqual(payload["base_tree"], "base-tree")
                self.assertEqual({row["path"] for row in payload["tree"]}, set(files))
                return {"sha": "new-tree"}
            if path.endswith("/commits"):
                self.assertIn(finish.TRAILER, payload["message"])
                self.assertEqual(payload["parents"], ["base-commit"])
                return {"sha": "new-commit"}
            if method == "PATCH":
                self.assertFalse(payload["force"])
                updated = True
                return {}
            if "trees/new-tree" in path:
                return {"tree": [
                    {"path": "unrelated.txt", "type": "blob", "sha": "keep"},
                    {"path": next(iter(files)), "type": "blob", "sha": digest},
                ]}
            if "/compare/" in path:
                return {"files": [{"filename": next(iter(files))}]}
            self.fail(path)

        with mock.patch.object(finish, "github", side_effect=api):
            self.assertEqual(finish.publish(dict(repo="owner/repo", branch="main"), files), "new-commit")
        self.assertEqual(sum(method == "PATCH" for method, _, _ in calls), 1)

    def test_pod_git_publication_preserves_unrelated_files_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origin = root / "origin.git"
            work = root / "initial"

            def git(*args):
                return subprocess.run(
                    ["git", *map(str, args)], check=True, capture_output=True, text=True,
                ).stdout.strip()

            git("init", "--bare", origin)
            git("init", work)
            git("-C", work, "checkout", "-b", "main")
            (work / "unrelated.txt").write_text("keep\n")
            git("-C", work, "add", "unrelated.txt")
            git("-C", work, "-c", "user.name=Test", "-c", "user.email=test@example.com",
                "commit", "-m", "base")
            git("-C", work, "push", origin, "HEAD:refs/heads/main")
            state_dir = root / "state"
            state_dir.mkdir()
            publication = state_dir / "publication.git"
            git("init", "--bare", publication)
            git("-C", publication, "remote", "add", "origin", origin)
            (state_dir / "checkpoint.pt").write_bytes(b"private")
            config = dict(repo="owner/repo", branch="main", pod_id="test-pod",
                          ssh_key="/unused-key", known_hosts="/unused-hosts")
            files = {"alpha_sweep/post_analysis.py": b"print('analysis')\n"}
            commit = finish.publish_git(config, state_dir, files)
            self.assertEqual(git("-C", origin, "show", "main:unrelated.txt"), "keep")
            self.assertEqual(git("-C", origin, "show", "main:alpha_sweep/post_analysis.py"), "print('analysis')")
            self.assertNotIn("checkpoint.pt", git("-C", origin, "ls-tree", "-r", "--name-only", "main"))
            self.assertIn(finish.TRAILER, git("-C", origin, "show", "-s", "--format=%B", commit))
            self.assertEqual(finish.publish_git(config, state_dir, files), commit)

    def test_pod_finalization_destroys_key_only_after_verified_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep = root / "sweep"
            output = sweep / "analysis"
            output.mkdir(parents=True)
            summary = self.summary()
            finish.atomic_json(sweep / "summary.json", summary)
            for name in finish.SOURCES:
                path = root / "sources" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"pass\n")
            manifest = dict(status="completed", files=[], summary_sha256=hashlib.sha256(
                (sweep / "summary.json").read_bytes()).hexdigest())
            for name in finish.OUTPUTS:
                content = b"fixture\n"
                (output / name).write_bytes(content)
                manifest["files"].append(dict(name=name, size=len(content),
                                              sha256=hashlib.sha256(content).hexdigest()))
            finish.atomic_json(output / "analysis_manifest.json", manifest)
            key = root / "key"
            config = dict(
                sweep_dir=str(sweep), expected_config=summary["config"], repo="owner/repo",
                results_relative="alpha_sweep/results/test", ssh_key=str(key),
            )
            for fails in (False, True):
                key.write_bytes(b"private fixture")
                state = {}
                events = []

                def publish(config, directory, files):
                    self.assertTrue(key.exists())
                    self.assertEqual(set(files), set(finish.SOURCES) | {
                        "alpha_sweep/results/test/" + name for name in finish.OUTPUTS
                    })
                    if fails:
                        raise RuntimeError("push failed")
                    events.append("verified push")
                    return "commit"

                def stop(*args):
                    self.assertFalse(key.exists())
                    self.assertEqual(state["published_commit"], "commit")
                    events.append("stop")

                with mock.patch.object(finish.subprocess, "run"), \
                     mock.patch.object(finish, "validate_publish_files"), \
                     mock.patch.object(finish, "publish_git", side_effect=publish), \
                     mock.patch.object(finish, "finish_shutdown", side_effect=stop) as shutdown:
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "push failed"):
                            finish.finalize_pod(config, root, "token", state)
                        shutdown.assert_not_called()
                        self.assertTrue(key.exists())
                    else:
                        finish.finalize_pod(config, root, "token", state)
                        self.assertEqual(events, ["verified push", "stop"])

    def test_pod_monitor_refuses_to_run_outside_tmux(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(finish, "read_token") as token:
            with self.assertRaisesRegex(RuntimeError, "inside tmux"):
                finish.monitor_pod({}, Path("/unused"), {})
            token.assert_not_called()

    def test_session_cleanup_checks_every_pane_in_the_target_session(self):
        config = dict(on_pod=True, training_session="alpha-sweep", analysis_session="alpha-post-analysis")

        def execute(command):
            exec(command[2], {"__name__": "__main__"})

        for panes, allowed in (("1\n1\n", True), ("1\n0\n", False), ("", False)):
            with self.subTest(panes=panes), \
                 mock.patch.object(finish, "run", side_effect=execute), \
                 mock.patch.object(finish.subprocess, "run") as action, \
                 mock.patch.object(finish.subprocess, "check_output", return_value=panes) as probe, \
                 mock.patch.object(sys, "argv", ["-c", json.dumps(config)]):
                action.return_value.returncode = 0
                if allowed:
                    finish.close_sessions(config)
                    self.assertEqual(sum(call.args[0][1] == "kill-session" for call in action.call_args_list), 2)
                else:
                    with self.assertRaisesRegex(RuntimeError, "live tmux"):
                        finish.close_sessions(config)
                    self.assertFalse(any(call.args[0][1] == "kill-session" for call in action.call_args_list))
                self.assertEqual(probe.call_args_list[0].args[0],
                                 ["tmux", "list-panes", "-s", "-t", "=alpha-sweep", "-F", "#{pane_dead}"])

    def test_shutdown_only_follows_verified_publication(self):
        summary = self.summary()
        config = dict(
            sweep_dir="/workspace/sweep", expected_config=summary["config"],
            repo="owner/repo", pod_id="test-pod",
        )
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            state = {}
            with mock.patch.object(finish, "remote_json", return_value=summary), \
                 mock.patch.object(finish, "start_analysis"), \
                 mock.patch.object(finish, "ssh", return_value="True"), \
                 mock.patch.object(finish, "fetch_analysis", return_value={}), \
                 mock.patch.object(finish, "validate_publish_files"), \
                 mock.patch.object(finish, "publish", side_effect=lambda *args: calls.append("published") or "commit"), \
                 mock.patch.object(finish, "close_sessions", side_effect=lambda *args: calls.append("closed")), \
                 mock.patch.object(finish, "runpod_request", side_effect=[
                     {"status": "RUNNING", "actions": ["stop"]},
                     None,
                     {"status": "EXITED"},
                 ]) as request, mock.patch("builtins.print"):
                finish.finalize(config, Path(directory), "token", state)
            self.assertEqual(calls, ["published", "closed"])
            self.assertEqual(request.call_args_list[1].args, ("token", "test-pod", "stop"))
            self.assertEqual(state["phase"], "completed")
            self.assertEqual(json.loads((Path(directory) / "state.json").read_text())["pod_status"], "EXITED")

            with mock.patch.object(finish, "remote_json", return_value=summary), \
                 mock.patch.object(finish, "start_analysis"), \
                 mock.patch.object(finish, "ssh", return_value="True"), \
                 mock.patch.object(finish, "fetch_analysis", return_value={}), \
                 mock.patch.object(finish, "validate_publish_files"), \
                 mock.patch.object(finish, "publish", side_effect=RuntimeError("push failed")), \
                 mock.patch.object(finish, "close_sessions") as close, \
                 mock.patch.object(finish, "runpod_request") as stop:
                with self.assertRaisesRegex(RuntimeError, "push failed"):
                    finish.finalize(config, Path(directory), "token", {})
                close.assert_not_called()
                stop.assert_not_called()

    def test_stop_verification_can_resume_without_ssh_after_pod_stops(self):
        state = dict(
            published_commit="commit", publication_url="https://github.com/owner/repo/commit/commit",
            sessions_closed=True, phase="stopping",
        )
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(finish, "close_sessions") as close, \
             mock.patch.object(finish, "runpod_request", return_value={"status": "EXITED"}), \
             mock.patch("builtins.print"):
            finish.finish_shutdown(dict(pod_id="test-pod"), Path(directory), "token", state)
            close.assert_not_called()
            self.assertEqual(state["phase"], "completed")

    def test_pod_watcher_only_runs_analysis_after_successful_completion(self):
        summary = self.summary()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            finish.atomic_json(path, summary)
            with mock.patch.object(finish.subprocess, "run") as analyzer, \
                 mock.patch("builtins.print"):
                finish.pod_watch(Path(directory), Path("/project/alpha_sweep/post_analysis.py"))
                self.assertEqual(analyzer.call_args.args[0][-2:], [
                    "--output-dir", str(Path(directory) / "analysis"),
                ])
                self.assertTrue(analyzer.call_args.kwargs["check"])
            summary["status"] = "failed"
            finish.atomic_json(path, summary)
            with mock.patch.object(finish.subprocess, "run") as analyzer:
                with self.assertRaisesRegex(RuntimeError, "not finish successfully"):
                    finish.pod_watch(Path(directory), Path("/project/alpha_sweep/post_analysis.py"))
                analyzer.assert_not_called()

    def test_monitor_verifies_shutdown_without_reconnecting_to_stopped_pod(self):
        state = dict(phase="stopping", published_commit="commit", sessions_closed=True)
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(finish, "read_token", return_value="token"), \
             mock.patch.object(finish, "remote_json") as remote, \
             mock.patch.object(finish, "finish_shutdown") as shutdown:
            finish.monitor(dict(pod_id="test-pod"), Path(directory), state)
            remote.assert_not_called()
            shutdown.assert_called_once()


class SweepLauncherTests(unittest.TestCase):
    def test_launcher_starts_real_sweep_and_finalizer_in_tmux_without_smoke_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = root / "calls.jsonl"
            python = root / "fake-python"
            python.write_text(
                "#!" + sys.executable + "\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "with open(os.environ['CALLS'], 'a') as stream:\n"
                " stream.write(json.dumps(['python'] + sys.argv[1:]) + '\\n')\n"
                "if '--sweep' in sys.argv:\n"
                " root = Path(sys.argv[sys.argv.index('--output-dir')+1])\n"
                " root.mkdir()\n"
                " (root/'summary.json').write_text('{}')\n"
            )
            python.chmod(0o700)
            tmux = root / "tmux"
            tmux.write_text(
                "#!" + sys.executable + "\n"
                "import json, os, subprocess, sys\n"
                "with open(os.environ['CALLS'], 'a') as stream:\n"
                " stream.write(json.dumps(['tmux'] + sys.argv[1:]) + '\\n')\n"
                "if sys.argv[1] == 'has-session': sys.exit(1)\n"
                "if sys.argv[1] == 'new-session':\n"
                " env = dict(os.environ, TMUX='test-session')\n"
                " sys.exit(subprocess.run(['bash','-c',sys.argv[-1]], env=env).returncode)\n"
            )
            tmux.chmod(0o700)
            environment = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"],
                               PYTHON=str(python), CALLS=str(calls), RUNPOD_POD_ID="test-pod")
            sweep = root / "results with spaces"
            command = [
                "bash", str(ROOT / "alpha_sweep/launch.sh"),
                str(sweep), "test-pod", "/dev/shm/key", "/dev/shm/hosts", "42",
            ]
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            records = [json.loads(line) for line in calls.read_text().splitlines()]
            sessions = [row[row.index("-s")+1] for row in records if row[:2] == ["tmux", "new-session"]]
            self.assertEqual(sessions, ["alpha-finalize", "alpha-sweep"])
            python_calls = [row for row in records if row[0] == "python"]
            self.assertEqual(len(python_calls), 3)
            self.assertIn("--sweep", python_calls[0])
            self.assertEqual(python_calls[0][python_calls[0].index("--finalist-steps")+1], "500")
            self.assertIn("--skip-preflight", python_calls[1])
            self.assertIn("--prepare", python_calls[1])
            self.assertNotIn("--prepare", python_calls[2])
            before = len(records)
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("never overwritten", result.stderr)
            self.assertEqual(len(calls.read_text().splitlines()), before)


if __name__ == "__main__":
    unittest.main()
