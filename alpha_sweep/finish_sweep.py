#!/usr/bin/env python3
"""Publish a completed sweep, then stop its pod.

Pod mode runs inside tmux with a repository-only deploy key kept in RAM.
Mac mode keeps GitHub authentication local. No API credentials are logged.
"""

import argparse
import base64
import fcntl
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request


STAGES = ("screening", "refinement", "confirmation")
SOURCES = (
    "alpha_sweep.md",
    "alpha_sweep/launch.sh",
    "alpha_sweep/models.py",
    "alpha_sweep/train_gpt2_cond_muon.py",
    "alpha_sweep/post_analysis.py",
    "alpha_sweep/finish_sweep.py",
    "tests/test_alpha_sweep.py",
    "tests/test_post_analysis.py",
    "tests/test_finish_sweep.py",
)
OUTPUTS = ("analysis.ipynb", "experiment_report.md")
TRAILER = "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"


class TransientError(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(command, input_text=None, timeout=60):
    try:
        completed = subprocess.run(
            command, input=input_text, text=True, capture_output=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, ConnectionError) as error:
        raise TransientError(f"{command[0]} timed out or disconnected") from error
    if completed.returncode:
        message = completed.stderr[-2000:]
        transient = completed.returncode == 255 and command[0] in ("ssh", "scp")
        transient = transient or any(text in message.lower() for text in (
            "http 429", "http 500", "http 502", "http 503", "http 504",
            "tls handshake timeout", "temporary failure", "connection reset",
            "could not resolve", "no such host", "network is unreachable",
            "connection timed out", "connection closed", "remote end hung up",
        ))
        exception = TransientError if transient else RuntimeError
        raise exception(f"{command[0]} exited {completed.returncode}: {message}")
    return completed.stdout


def ssh(config, command):
    return run([
        "ssh", "-i", config["ssh_key"], "-p", str(config["port"]),
        "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "UserKnownHostsFile=" + config["known_hosts"],
        "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3", "root@" + config["host"],
        shlex.join(command),
    ])


def remote_json(config, path):
    return json.loads(ssh(config, ["cat", path]))


def runpod_request(token, pod_id, action=None):
    url = "https://api.runpod.io/v2/pods/" + pod_id
    data = None
    if action is not None:
        if action != "stop":
            raise ValueError("Only the stop action is permitted")
        url += "/action"
        data = json.dumps({"action": "stop"}).encode()
    request = urllib.request.Request(
        url, data=data,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as error:
        if error.code == 429 or error.code >= 500:
            raise TransientError(f"Runpod HTTP {error.code}") from error
        raise RuntimeError(f"Runpod HTTP {error.code}; no alternate credentials will be tried") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise TransientError("Runpod request timed out or disconnected") from error


def validate_summary(summary, expected_config):
    if summary["config"] != expected_config:
        raise ValueError("Sweep configuration changed after finalization was armed")
    if summary["status"] not in ("completed", "completed_with_divergences"):
        raise ValueError("Analysis and publication require a successfully completed sweep")
    if "active_trial" in summary:
        raise ValueError("A completed sweep still has an active trial")
    rounds = expected_config.get("rounds", 3)
    if type(rounds) is not int or rounds < 1:
        raise ValueError("Sweep rounds must be a positive integer")
    stages = STAGES if rounds == 3 else tuple(f"round_{index}" for index in range(1, rounds + 1))
    grids = summary.get("round_grids", {})
    if not isinstance(grids, dict) or set(grids) != set(stages):
        raise ValueError("Sweep round grids are incomplete or unexpected")
    stage_keys = {key for key in summary if key in STAGES or re.fullmatch(r"round_[0-9]+", key)}
    if stage_keys != set(stages):
        raise ValueError("Sweep stages are incomplete or unexpected")
    if "alpha_grid" in expected_config and grids[stages[0]] != expected_config["alpha_grid"]:
        raise ValueError("Initial round grid does not match the sweep configuration")
    for stage in stages:
        seeds = (
            expected_config["finalist_seeds"] if stage == stages[-1]
            else [expected_config["screening_seed"]]
        )
        grid = grids[stage]
        if not grid or len(set(grid)) != len(grid) or not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"{stage} has an empty or duplicate grid or seed list")
        expected = {(alpha, seed) for alpha in grid for seed in seeds}
        actual = [(row["alpha"], row["seed"]) for row in summary[stage]]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError(f"{stage} has incomplete or duplicate trials")
        for row in summary[stage]:
            if row["status"] not in ("completed", "diverged"):
                raise ValueError(f"Unexpected trial status in {stage}")
            if row["status"] == "completed" and not math.isfinite(row["final_val_loss"]):
                raise ValueError(f"Nonfinite completed trial in {stage}")
    if (not summary["ranking"] or summary["best_alpha"] != summary["ranking"][0]["alpha"]
            or summary["best_alpha"] not in grids[stages[-1]]):
        raise ValueError("Completed sweep has no consistent final ranking")


def read_token(config, verify=True):
    # sshd drops the container's inherited environment.
    script = """
import os
import sys
from pathlib import Path
token = os.environ.get("RUNPOD_API_KEY")
if not token:
    for item in Path("/proc/1/environ").read_bytes().split(b"\\0"):
        if item.startswith(b"RUNPOD_API_KEY="):
            token = item.split(b"=", 1)[1].decode()
            break
if not token:
    raise RuntimeError("No Runpod credential in the pod process environment")
sys.stdout.write(token)
"""
    if config.get("on_pod"):
        token = os.environ.get("RUNPOD_API_KEY")
        if not token:
            token = next((
                item.split(b"=", 1)[1].decode()
                for item in Path("/proc/1/environ").read_bytes().split(b"\0")
                if item.startswith(b"RUNPOD_API_KEY=")
            ), None)
    elif config.get("env_file"):
        from dotenv import dotenv_values
        token = dotenv_values(config["env_file"], interpolate=False).get("RUNPOD_API_KEY")
        if not isinstance(token, str):
            raise RuntimeError("RUNPOD_API_KEY is missing from the selected environment file")
        token = token.strip()
    else:
        token = ssh(config, [
            "/workspace/venv/bin/python", "-c", script,
        ]).strip()
    if not isinstance(token, str) or not token or any(char.isspace() for char in token):
        raise RuntimeError("The selected source has no usable Runpod API credential")
    if verify:
        pod = runpod_request(token, config["pod_id"])
        if pod["id"] != config["pod_id"]:
            raise ValueError("Runpod authentication resolved a different pod")
    return token


def github(method, path, payload=None):
    command = ["gh", "api", "--method", method, path]
    if payload is not None:
        command.extend(["--input", "-"])
    response = run(command, None if payload is None else json.dumps(payload))
    return json.loads(response) if response.strip() else None


def validate_publish_files(files, token):
    secrets = re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")
    for path, content in files.items():
        if Path(path).suffix not in (".py", ".ipynb", ".md", ".sh"):
            raise ValueError(f"Not an allowed publication type: {path}")
        if len(content) > 8 * 1024 * 1024 or b"\0" in content:
            raise ValueError(f"Publication file is binary or exceeds 8 MiB: {path}")
        content.decode("utf-8")
        if token.encode() in content or secrets.search(content):
            raise ValueError(f"Publication file contains a credential: {path}")
        if path.endswith(".py"):
            compile(content, path, "exec")
        if path.endswith(".ipynb"):
            notebook = json.loads(content)
            if notebook["nbformat"] != 4:
                raise ValueError("Expected a version 4 notebook")
            images = 0
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), path, "exec")
                for output in cell.get("outputs", []):
                    if output["output_type"] != "display_data":
                        raise ValueError("Unexpected notebook output type")
                    if set(output["data"]) - {"image/png", "text/plain"}:
                        raise ValueError("Unexpected notebook MIME data")
                    if "image/png" in output["data"]:
                        image = base64.b64decode(output["data"]["image/png"], validate=True)
                        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
                            raise ValueError("Notebook image is not a PNG")
                        images += 1
            if images < 5:
                raise ValueError("Notebook is missing requested rendered plots")
        if path.endswith("experiment_report.md"):
            if len([part for part in content.decode().strip().split("\n\n") if part.strip()]) != 2:
                raise ValueError("Experiment report must contain exactly two paragraphs")


def blob_sha(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def git_command(config, directory, *arguments, input_text=None):
    identity = shlex.join([
        "ssh", "-i", config["ssh_key"], "-o", "IdentitiesOnly=yes",
        "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "UserKnownHostsFile=" + config["known_hosts"],
        "-o", "ConnectTimeout=15",
    ])
    owner = config["repo"].split("/")[0]
    return run([
        "git", "-c", "core.sshCommand=" + identity,
        "-c", "user.name=" + owner,
        "-c", "user.email=" + owner + "@users.noreply.github.com",
        "-C", str(directory), *arguments,
    ], input_text=input_text, timeout=120).strip()


def git_tree(config, directory, revision):
    output = git_command(config, directory, "ls-tree", "-r", "-z", revision)
    entries = {}
    for row in output.split("\0"):
        if row:
            metadata, path = row.split("\t", 1)
            mode, kind, digest = metadata.split()
            entries[path] = (mode, kind, digest)
    return entries


def publish_git(config, state_dir, files):
    directory = Path(state_dir) / "publication.git"
    if not directory.exists():
        run(["git", "init", "--bare", str(directory)])
        git_command(config, directory, "remote", "add", "origin", "git@github.com:" + config["repo"] + ".git")
    branch = "refs/heads/" + config["branch"]
    tracking = "refs/remotes/origin/" + config["branch"]
    for attempt in range(3):
        git_command(config, directory, "fetch", "--no-tags", "--depth=1", "origin", branch + ":" + tracking)
        head = git_command(config, directory, "rev-parse", tracking)
        existing = git_tree(config, directory, head)
        changed = {
            path: content for path, content in files.items()
            if existing.get(path) != ("100644", "blob", blob_sha(content))
        }
        if not changed:
            return head
        git_command(config, directory, "read-tree", head)
        for path, content in changed.items():
            digest = git_command(config, directory, "hash-object", "-w", "--stdin", input_text=content.decode())
            if digest != blob_sha(content):
                raise ValueError("Git blob hash did not match the publication file")
            git_command(config, directory, "update-index", "--add", "--cacheinfo", "100644," + digest + "," + path)
        tree = git_command(config, directory, "write-tree")
        message = "Add three-round alpha sweep analysis (" + config["pod_id"] + ")\n\n" + TRAILER + "\n"
        commit = git_command(config, directory, "commit-tree", tree, "-p", head, input_text=message)
        changed_paths = git_command(config, directory, "diff-tree", "--no-commit-id", "--name-only", "-r", head, commit)
        if set(changed_paths.splitlines()) != set(changed):
            raise ValueError("Publication commit contains unexpected paths")
        try:
            git_command(config, directory, "push", "origin", commit + ":" + branch)
        except RuntimeError as error:
            latest = git_command(config, directory, "ls-remote", "origin", branch).split()[0]
            if latest != head and attempt < 2:
                continue
            raise error
        git_command(config, directory, "fetch", "--no-tags", "origin", branch + ":" + tracking)
        latest = git_command(config, directory, "rev-parse", tracking)
        git_command(config, directory, "merge-base", "--is-ancestor", commit, latest)
        actual = git_tree(config, directory, latest)
        if any(actual.get(path) != ("100644", "blob", blob_sha(content)) for path, content in files.items()):
            raise ValueError("Published files failed fresh GitHub fetch verification")
        return commit
    raise RuntimeError("GitHub branch changed repeatedly; nothing was force-pushed")


def publish(config, files):
    repo = config["repo"]
    branch = config["branch"]
    prefix = "repos/" + repo + "/git/"
    for attempt in range(3):
        head = github("GET", prefix + "ref/heads/" + branch)["object"]["sha"]
        commit = github("GET", prefix + "commits/" + head)
        tree = github("GET", prefix + "trees/" + commit["tree"]["sha"] + "?recursive=1")
        if tree.get("truncated"):
            raise ValueError("GitHub tree was truncated; cannot verify publication paths")
        existing = {row["path"]: row["sha"] for row in tree["tree"] if row["type"] == "blob"}
        changed = {path: content for path, content in files.items() if existing.get(path) != blob_sha(content)}
        if not changed:
            return head
        entries = []
        for path, content in changed.items():
            blob = github("POST", prefix + "blobs", {
                "encoding": "base64", "content": base64.b64encode(content).decode(),
            })
            if blob["sha"] != blob_sha(content):
                raise ValueError("GitHub blob hash did not match the publication file")
            entries.append(dict(path=path, mode="100644", type="blob", sha=blob["sha"]))
        new_tree = github("POST", prefix + "trees", {
            "base_tree": commit["tree"]["sha"], "tree": entries,
        })
        created = github("POST", prefix + "commits", {
            "message": "Add three-round alpha sweep analysis and automation\n\n" + TRAILER,
            "parents": [head], "tree": new_tree["sha"],
        })
        try:
            github("PATCH", prefix + "refs/heads/" + branch, {"sha": created["sha"], "force": False})
        except RuntimeError as error:
            if "HTTP 422" in str(error) and attempt < 2:
                refreshed = github("GET", prefix + "ref/heads/" + branch)["object"]["sha"]
                if refreshed != head:
                    continue
            raise
        latest = github("GET", prefix + "ref/heads/" + branch)["object"]["sha"]
        if latest != created["sha"]:
            ancestry = github("GET", f"repos/{repo}/compare/{created['sha']}...{latest}")
            if ancestry["status"] not in ("ahead", "identical"):
                raise ValueError("Publication is no longer on the requested GitHub branch")
        latest_commit = github("GET", prefix + "commits/" + latest)
        actual = github("GET", prefix + "trees/" + latest_commit["tree"]["sha"] + "?recursive=1")
        if actual.get("truncated"):
            raise ValueError("Published GitHub tree was truncated")
        actual_hashes = {row["path"]: row["sha"] for row in actual["tree"] if row["type"] == "blob"}
        if any(actual_hashes.get(path) != blob_sha(content) for path, content in files.items()):
            raise ValueError("Published files failed GitHub hash verification")
        comparison = github("GET", f"repos/{repo}/compare/{head}...{created['sha']}")
        if {row["filename"] for row in comparison["files"]} != set(changed):
            raise ValueError("Published commit contains unexpected paths")
        return created["sha"]
    raise RuntimeError("GitHub branch changed repeatedly; nothing was force-pushed")


def start_analysis(config):
    code = (
        "import json,pathlib,shlex,subprocess,sys; "
        "c=json.loads(sys.argv[1]); "
        "manifest=pathlib.Path(c['sweep_dir'])/'analysis'/'analysis_manifest.json'; "
        "session=c['analysis_session']; "
        "found=subprocess.run(['tmux','has-session','-t','='+session],capture_output=True); "
        "command=shlex.join(['/workspace/venv/bin/python',c['remote_project']+'/alpha_sweep/post_analysis.py',"
        "'--sweep-dir',c['sweep_dir'],'--output-dir',c['sweep_dir']+'/analysis']); "
        "command+=' > '+shlex.quote(c['sweep_dir']+'/post-analysis.log')+' 2>&1'; "
        "\nif not manifest.exists() and found.returncode!=0:\n"
        " subprocess.run(['tmux','new-session','-d','-s',session,'-c',c['remote_project'],command],check=True)\n"
        " subprocess.run(['tmux','set-window-option','-t','='+session,'remain-on-exit','on'],check=True)\n"
    )
    ssh(config, ["/workspace/venv/bin/python", "-c", code, json.dumps(config)])


def fetch_analysis(config, state_dir):
    remote = config["sweep_dir"] + "/analysis"
    manifest = remote_json(config, remote + "/analysis_manifest.json")
    if manifest["status"] != "completed" or {entry["name"] for entry in manifest["files"]} != set(OUTPUTS):
        raise ValueError("Analysis manifest is incomplete or contains unexpected outputs")
    summary_bytes = ssh(config, ["cat", config["sweep_dir"] + "/summary.json"]).encode()
    if hashlib.sha256(summary_bytes).hexdigest() != manifest["summary_sha256"]:
        raise ValueError("Analysis belongs to a different sweep summary")
    destination = Path(config["project_root"]) / config["results_relative"]
    destination.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        name = entry["name"]
        temporary = state_dir / (name + ".download")
        run([
            "scp", "-i", config["ssh_key"], "-P", str(config["port"]),
            "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", "UserKnownHostsFile=" + config["known_hosts"],
            "-o", "ConnectTimeout=15",
            "root@" + config["host"] + ":" + remote + "/" + name, str(temporary),
        ], timeout=120)
        content = temporary.read_bytes()
        if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Downloaded {name} failed checksum verification")
        target = destination / name
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite an unrelated local result: {target}")
        temporary.replace(target)
    files = {
        name: (state_dir / "sources" / name).read_bytes() for name in SOURCES
    }
    files.update({
        config["results_relative"] + "/" + name: (destination / name).read_bytes()
        for name in OUTPUTS
    })
    return files


def close_sessions(config):
    code = (
        "import json,subprocess,sys; c=json.loads(sys.argv[1]); "
        "\nfor name in (c['training_session'],c['analysis_session']):\n"
        " found=subprocess.run(['tmux','has-session','-t','='+name],capture_output=True)\n"
        " if found.returncode==0:\n"
        "  dead=subprocess.check_output(['tmux','list-panes','-s','-t','='+name,'-F','#{pane_dead}'],text=True).splitlines()\n"
        "  if not dead or any(value!='1' for value in dead): raise RuntimeError('Refusing to close a live tmux process: '+name)\n"
        "  subprocess.run(['tmux','kill-session','-t','='+name],check=True)\n"
    )
    command = ["/workspace/venv/bin/python", "-c", code, json.dumps(config)]
    if config.get("on_pod"):
        run(command)
    else:
        ssh(config, command)


def finish_shutdown(config, state_dir, token, state):
    if not state.get("sessions_closed"):
        close_sessions(config)
        state["sessions_closed"] = True
    state["phase"] = "stopping"
    atomic_json(state_dir / "state.json", state)
    pod = runpod_request(token, config["pod_id"])
    if pod["status"] != "EXITED":
        if "stop" not in pod["actions"]:
            raise RuntimeError("The pod cannot be stopped in its current state")
        print("Stopping compute; container disk is cleared, retained volumes continue billing.", flush=True)
        runpod_request(token, config["pod_id"], "stop")
    for _ in range(40):
        pod = runpod_request(token, config["pod_id"])
        if pod["status"] == "EXITED":
            state["phase"] = "completed"
            state["pod_status"] = "EXITED"
            atomic_json(state_dir / "state.json", state)
            print("Published, tmux closed, and pod stop verified: " + state["publication_url"], flush=True)
            return
        time.sleep(3)
    raise RuntimeError("Pod did not reach EXITED after the stop request")


def finalize(config, state_dir, token, state):
    summary = remote_json(config, config["sweep_dir"] + "/summary.json")
    validate_summary(summary, config["expected_config"])
    if not state.get("published_commit"):
        start_analysis(config)
        while True:
            probe = ssh(config, [
                "/workspace/venv/bin/python", "-c",
                "import pathlib,sys; print(pathlib.Path(sys.argv[1]).is_file())",
                config["sweep_dir"] + "/analysis/analysis_manifest.json",
            ]).strip()
            if probe == "True":
                break
            dead = ssh(config, [
                "tmux", "display-message", "-p", "-t",
                "=" + config["analysis_session"], "#{pane_dead}",
            ]).strip()
            if dead == "1":
                raise RuntimeError("Post-analysis failed; inspect " + config["sweep_dir"] + "/post-analysis.log")
            time.sleep(5)
        files = fetch_analysis(config, state_dir)
        validate_publish_files(files, token)
        state["published_commit"] = publish(config, files)
        state["publication_url"] = f"https://github.com/{config['repo']}/commit/{state['published_commit']}"
        state["phase"] = "published"
        atomic_json(state_dir / "state.json", state)
    finish_shutdown(config, state_dir, token, state)


def prepare(options):
    root = options.project_root.resolve()
    state_dir = options.state_dir.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", options.repo):
        raise ValueError("Expected owner/repository")
    ipaddress.ip_address(options.host)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", options.pod_id):
        raise ValueError("Invalid pod ID")
    if not 1 <= options.port <= 65535:
        raise ValueError("Invalid SSH port")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", options.branch):
        raise ValueError("Use a simple branch name without slashes")
    relative = Path(options.results_relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Results destination must be inside the repository")
    for path in (options.sweep_dir, options.remote_project):
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", path) or ".." in Path(path).parts:
            raise ValueError("Expected a simple absolute pod path")
    config = dict(
        project_root=str(root), host=options.host, port=options.port,
        pod_id=options.pod_id, ssh_key=str(options.ssh_key.expanduser().resolve()),
        known_hosts=str(options.known_hosts.expanduser().resolve()),
        repo=options.repo, branch=options.branch, results_relative=str(relative),
        sweep_dir=options.sweep_dir, remote_project=options.remote_project,
        training_session="alpha-sweep", analysis_session="alpha-post-analysis",
        env_file=str(options.env_file.expanduser().resolve()) if options.env_file else None,
    )
    token = read_token(config)
    summary = remote_json(config, config["sweep_dir"] + "/summary.json")
    if summary["status"] == "failed":
        raise RuntimeError("Cannot arm finalization for a failed sweep")
    config["expected_config"] = summary["config"]
    account = github("GET", "repos/" + config["repo"])
    if not account.get("permissions", {}).get("push"):
        raise RuntimeError("GitHub authentication cannot push to the requested repository")
    files = {name: (root / name).read_bytes() for name in SOURCES}
    validate_publish_files(files, token)
    deployed = ssh(config, ["sha256sum", config["remote_project"] + "/alpha_sweep/train_gpt2_cond_muon.py"]).split()[0]
    if deployed != hashlib.sha256(files["alpha_sweep/train_gpt2_cond_muon.py"]).hexdigest():
        raise ValueError("Local trainer does not match the running experiment")
    state_dir.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        target = state_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    atomic_json(state_dir / "config.json", config)
    atomic_json(state_dir / "state.json", dict(phase="waiting"))
    print("Finalizer preflight passed. No credentials were persisted.", flush=True)


def pod_watch(sweep_dir, analyzer):
    summary_path = Path(sweep_dir) / "summary.json"
    expected_config = json.loads(summary_path.read_text())["config"]
    while True:
        summary = json.loads(summary_path.read_text())
        if summary["config"] != expected_config:
            raise ValueError("Sweep configuration changed while waiting for analysis")
        if summary["status"] in ("completed", "completed_with_divergences"):
            validate_summary(summary, expected_config)
            subprocess.run([
                sys.executable, str(analyzer), "--sweep-dir", str(sweep_dir),
                "--output-dir", str(Path(sweep_dir) / "analysis"),
            ], check=True)
            print("Post-analysis complete; waiting for publication from the Mac.", flush=True)
            return
        if summary["status"] != "running":
            raise RuntimeError("Sweep did not finish successfully; post-analysis was not run")
        print("Waiting on the pod for all sweep rounds.", flush=True)
        time.sleep(30)


def prepare_pod(options):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", options.repo or ""):
        raise ValueError("Expected owner/repository")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", options.branch):
        raise ValueError("Use a simple branch name without slashes")
    relative = Path(options.results_relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Results destination must be inside the repository")
    key = options.ssh_key.resolve()
    if Path("/dev/shm") not in key.parents or key.stat().st_mode & 0o077:
        raise ValueError("The deploy key must be owner-only and stored in pod RAM under /dev/shm")
    config = dict(
        on_pod=True, pod_id=options.pod_id, repo=options.repo, branch=options.branch,
        project_root=str(options.project_root.resolve()), sweep_dir=options.sweep_dir,
        results_relative=str(relative), ssh_key=str(key),
        known_hosts=str(options.known_hosts.resolve()), deploy_key_id=options.deploy_key_id,
        training_session="alpha-sweep", analysis_session="alpha-post-analysis",
        finalization_session="alpha-finalize",
    )
    token = read_token(config, verify=not options.skip_preflight)
    summary = json.loads((Path(config["sweep_dir"]) / "summary.json").read_text())
    if summary["status"] != "running":
        raise ValueError("Arm pod-side finalization while the sweep is running")
    config["expected_config"] = summary["config"]
    files = {name: (options.project_root / name).read_bytes() for name in SOURCES}
    validate_publish_files(files, token)
    state_dir = options.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        target = state_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    if not options.skip_preflight:
        directory = state_dir / "publication.git"
        run(["git", "init", "--bare", str(directory)])
        git_command(config, directory, "remote", "add", "origin", "git@github.com:" + config["repo"] + ".git")
        branch = "refs/heads/" + config["branch"]
        git_command(config, directory, "fetch", "--no-tags", "--depth=1", "origin", branch)
        head = git_command(config, directory, "rev-parse", "FETCH_HEAD")
        git_command(config, directory, "push", "--dry-run", "origin", head + ":" + branch)
    atomic_json(state_dir / "config.json", config)
    atomic_json(state_dir / "state.json", dict(phase="waiting"))
    if options.skip_preflight:
        print("Sources frozen; preflight API probe and Git push dry-run skipped by request.", flush=True)
    else:
        print("Pod preflight passed: repository-only Git push and Runpod access verified.", flush=True)


def finalize_pod(config, state_dir, token, state):
    if not state.get("published_commit"):
        sweep_dir = Path(config["sweep_dir"])
        summary_bytes = (sweep_dir / "summary.json").read_bytes()
        validate_summary(json.loads(summary_bytes), config["expected_config"])
        subprocess.run([
            sys.executable, str(state_dir / "sources/alpha_sweep/post_analysis.py"),
            "--sweep-dir", str(sweep_dir),
        ], check=True)
        output = sweep_dir / "analysis"
        manifest = json.loads((output / "analysis_manifest.json").read_text())
        if (manifest["status"] != "completed" or
                {entry["name"] for entry in manifest["files"]} != set(OUTPUTS) or
                len(manifest["files"]) != len(OUTPUTS) or
                manifest["summary_sha256"] != hashlib.sha256(summary_bytes).hexdigest()):
            raise ValueError("Analysis manifest does not match the completed sweep")
        files = {name: (state_dir / "sources" / name).read_bytes() for name in SOURCES}
        for entry in manifest["files"]:
            content = (output / entry["name"]).read_bytes()
            if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError("Analysis output failed checksum verification")
            files[config["results_relative"] + "/" + entry["name"]] = content
        validate_publish_files(files, token)
        state["published_commit"] = publish_git(config, state_dir, files)
        state["publication_url"] = f"https://github.com/{config['repo']}/commit/{state['published_commit']}"
        state["phase"] = "published"
        atomic_json(state_dir / "state.json", state)
    for path in (Path(config["ssh_key"]), Path(config["ssh_key"] + ".pub")):
        path.unlink(missing_ok=True)
    state["deploy_key_destroyed"] = True
    atomic_json(state_dir / "state.json", state)
    finish_shutdown(config, state_dir, token, state)


def monitor_pod(config, state_dir, state):
    if not os.environ.get("TMUX"):
        raise RuntimeError("Pod publication and shutdown must run inside tmux")
    session = run(["tmux", "display-message", "-p", "#{session_name}"]).strip()
    if session != config["finalization_session"]:
        raise RuntimeError("Run pod finalization in its configured tmux session")
    token = read_token(config)
    while True:
        try:
            if state.get("published_commit"):
                finalize_pod(config, state_dir, token, state)
                return
            summary = json.loads((Path(config["sweep_dir"]) / "summary.json").read_text())
            if summary["config"] != config["expected_config"]:
                raise ValueError("Sweep configuration changed")
            if summary["status"] in ("completed", "completed_with_divergences"):
                finalize_pod(config, state_dir, token, state)
                return
            if summary["status"] != "running":
                raise RuntimeError("Sweep failed; publication and shutdown were not performed")
            state.update(phase="waiting", last_poll=time.time(), active_trial=summary.get("active_trial"))
            state.pop("last_connection_error", None)
            atomic_json(state_dir / "state.json", state)
            active = summary.get("active_trial", {})
            print(f"Waiting in pod tmux: {active.get('stage')}, alpha={active.get('alpha')}", flush=True)
        except TransientError as error:
            state.update(last_connection_error=str(error), last_poll=time.time())
            atomic_json(state_dir / "state.json", state)
            print("Temporary connection failure; retrying: " + str(error), flush=True)
        except (RuntimeError, ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
            state.update(phase="failed", error=str(error))
            atomic_json(state_dir / "state.json", state)
            raise
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--pod-watch", action="store_true", help="on the pod, wait for the sweep then generate analysis")
    parser.add_argument("--on-pod", action="store_true", help="publish and stop from pod tmux using a RAM-only deploy key")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="with --on-pod --prepare, skip API probe and Git dry-run; retain runtime safeguards")
    parser.add_argument("--deploy-key-id", type=int, help="GitHub deploy key ID, recorded for later public-key cleanup")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--pod-id")
    parser.add_argument("--ssh-key", type=Path, default=Path("~/.ssh/id_ed25519"))
    parser.add_argument("--known-hosts", type=Path)
    parser.add_argument("--env-file", type=Path, help="read RUNPOD_API_KEY from this local file without executing it")
    parser.add_argument("--repo")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--sweep-dir", default="/workspace/alpha-sweep-3-rounds")
    parser.add_argument("--remote-project", default="/workspace/condition-number-adaptive-muon")
    parser.add_argument("--results-relative", default="alpha_sweep/results/three-rounds")
    options = parser.parse_args()
    if options.skip_preflight and not (options.on_pod and options.prepare):
        parser.error("--skip-preflight requires --on-pod --prepare")
    if options.pod_watch:
        pod_watch(
            options.sweep_dir,
            Path(options.remote_project) / "alpha_sweep" / "post_analysis.py",
        )
        return
    if options.state_dir is None:
        parser.error("--state-dir is required for finalization")
    if options.prepare:
        if options.on_pod:
            if not all((options.pod_id, options.known_hosts, options.repo, options.deploy_key_id)):
                parser.error("--on-pod --prepare needs pod-id, known-hosts, repo and deploy-key-id")
            prepare_pod(options)
            return
        if not all((options.host, options.port, options.pod_id, options.known_hosts, options.repo)):
            parser.error("--prepare needs host, port, pod-id, known-hosts and repo")
        prepare(options)
        return
    state_dir = options.state_dir.resolve()
    config = json.loads((state_dir / "config.json").read_text())
    state = json.loads((state_dir / "state.json").read_text())
    if state["phase"] == "completed":
        print("Already completed: " + state["publication_url"], flush=True)
        return
    with (state_dir / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A finalizer is already monitoring this sweep") from error
        state["worker_pid"] = os.getpid()
        if config.get("on_pod"):
            monitor_pod(config, state_dir, state)
        else:
            monitor(config, state_dir, state)


def monitor(config, state_dir, state):
    token = None
    try:
        while True:
            try:
                if token is None:
                    token = read_token(config)
                if state.get("published_commit") and state.get("sessions_closed"):
                    finish_shutdown(config, state_dir, token, state)
                    return
                summary = remote_json(config, config["sweep_dir"] + "/summary.json")
                if summary["config"] != config["expected_config"]:
                    raise ValueError("Sweep configuration changed")
                if summary["status"] == "failed":
                    raise RuntimeError("Sweep failed; publication and shutdown were not performed")
                if summary["status"] in ("completed", "completed_with_divergences"):
                    finalize(config, state_dir, token, state)
                    return
                if summary["status"] != "running":
                    raise ValueError("Unknown sweep status")
                state.update(phase="waiting", last_poll=time.time(), active_trial=summary.get("active_trial"))
                state.pop("last_connection_error", None)
                atomic_json(state_dir / "state.json", state)
                active = summary.get("active_trial", {})
                print(f"Waiting: {active.get('stage')}, alpha={active.get('alpha')}", flush=True)
            except TransientError as error:
                state.update(last_connection_error=str(error), last_poll=time.time())
                atomic_json(state_dir / "state.json", state)
                print("Temporary connection failure; retrying: " + str(error), flush=True)
            time.sleep(60)
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        state.update(phase="failed", error=str(error))
        atomic_json(state_dir / "state.json", state)
        raise


if __name__ == "__main__":
    main()
