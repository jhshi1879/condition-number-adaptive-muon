#!/usr/bin/env python3
"""DEPRECATED: this launcher is no longer maintained and may be removed.

Start an existing stopped Runpod pod, train, and stop the pod.

Requires Python 3.10+, dependencies in requirements-runpod.txt, ssh/scp, and a
public SSH key registered with Runpod. Set RUNPOD_API_KEY and RUNPOD_POD_ID in
the adjacent .env file or the environment. Environment variables take precedence.
The pod must expose TCP port 22. A persistent volume is not required.
On the first connection, verify the host fingerprint before accepting it.

Example:
    python runpod_launch.py --ssh-key ~/.ssh/runpod_ed25519

By default this uses one GPU and downloads one training shard. --full-run uses
eight GPUs and downloads the complete FineWeb10B dataset. Both modes use the
trainer's configured hyperparameters, start from scratch, and cannot resume.
Logs and checkpoints are copied to runpod-results on this Mac before stopping.
The pod's Python must support venv and have CUDA-enabled PyTorch and NumPy.
The dataset downloader runs in an isolated virtual environment.
The pod is stopped even if copying fails; files on its container disk are lost.
Stopping is best effort: power loss, SIGKILL, or network failure can leave the
pod running and accruing charges.
"""

import argparse
import ipaddress
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid


HERE = Path(__file__).resolve().parent
ENV_FILE = HERE / ".env"
TRAINER = HERE / "train_gpt2.py"
DOWNLOADER = HERE / "data" / "cached_fineweb10B.py"
WORKSPACE = "/workspace"
RESULTS_DIR = HERE / "runpod-results"

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-key", type=Path, required=True, help="SSH private key")
    parser.add_argument("--full-run", action="store_true",
                        help="use eight GPUs and download the complete dataset")
    parser.add_argument("--startup-timeout", type=int, default=600, help="seconds")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR,
                        help="local directory for logs and checkpoints")
    args = parser.parse_args(argv)
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive")
    return args


def get_pod(runpod, pod_id):
    pod = runpod.get_pod(pod_id)
    if not isinstance(pod, dict):
        raise RuntimeError("Pod not found: " + pod_id)
    if "status" not in pod or "mounts" not in pod:
        raise RuntimeError("Runpod returned an unexpected pod response; upgrade the runpod SDK to its v2 API version")
    return pod


def ssh_endpoint(pod):
    direct = (pod.get("ssh") or {}).get("direct") or {}
    host, port, user = direct.get("host"), direct.get("port"), direct.get("username")
    if not host or not port or not user:
        return None
    try:
        if ipaddress.ip_address(host).version != 4:
            raise RuntimeError("Only IPv4 direct SSH addresses are supported")
    except ValueError as exc:
        raise RuntimeError("Unexpected non-IP direct SSH host in pod response") from exc
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise RuntimeError("Invalid SSH port in pod response")
    if user != "root":
        raise RuntimeError("Unexpected direct SSH user; this launcher expects root")
    return host, port, user


def wait_for_ssh(runpod, pod_id, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pod = get_pod(runpod, pod_id)
        status = pod["status"]
        if status in ("ERROR", "TERMINATED", "EXITED"):
            raise RuntimeError("Pod entered unexpected state: " + status)
        endpoint = ssh_endpoint(pod) if status == "RUNNING" else None
        if endpoint:
            host, port, _ = endpoint
            try:
                with socket.create_connection((host, port), timeout=5):
                    return endpoint
            except OSError:
                pass
        time.sleep(5)
    raise TimeoutError("Timed out waiting for the pod's direct SSH port")


def ssh_options(key, port):
    return [
        "-i", str(key), "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=ask",
        "-o", "ConnectTimeout=15", "-p", str(port),
    ]


def remote(ssh, command):
    subprocess.run(ssh + [command], check=True)


def scp(key, host, port, user, source, destination):
    options = ssh_options(key, port)
    options[options.index("-p")] = "-P"
    subprocess.run(["scp"] + options + [str(source), f"{user}@{host}:{destination}"], check=True)


def copy_results(key, host, port, user, run_dir, local_dir):
    options = ssh_options(key, port)
    options[options.index("-p")] = "-P"
    sources = [f"{user}@{host}:{run_dir}/training.log", f"{user}@{host}:{run_dir}/logs"]
    subprocess.run(["scp", "-r"] + options + sources + [str(local_dir)], check=True)


def verify_stopped(runpod, pod_id):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        pod = get_pod(runpod, pod_id)
        if pod["status"] == "EXITED":
            return
        if pod["status"] == "TERMINATED":
            raise RuntimeError("Pod was terminated rather than stopped")
        time.sleep(5)
    raise TimeoutError("Pod did not reach EXITED within 120 seconds")


def stop_pod(runpod, pod_id):
    try:
        runpod.stop_pod(pod_id)
    except runpod.error.QueryError as exc:
        if ('action "stop" is not valid for status "EXITED"' not in str(exc)
                or get_pod(runpod, pod_id)["status"] != "EXITED"):
            raise
        print("Pod is already stopped.", flush=True)
        return
    verify_stopped(runpod, pod_id)


def launch(args, runpod):
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id or not os.environ.get("RUNPOD_API_KEY"):
        raise RuntimeError(f"Set RUNPOD_POD_ID and RUNPOD_API_KEY in {ENV_FILE} or your environment")
    key = args.ssh_key.expanduser().resolve()
    if not key.is_file():
        raise RuntimeError(f"SSH private key not found: {key}")
    for path in (TRAINER, DOWNLOADER):
        if not path.is_file():
            raise RuntimeError(f"Required file not found: {path}")
    for command in ("ssh", "scp"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Required command not found: {command}")

    pod = get_pod(runpod, pod_id)
    if pod["status"] != "EXITED":
        raise RuntimeError(f"Pod must be stopped before launch; current status: {pod['status']}")
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    run_dir = f"{WORKSPACE}/muon-runs/{run_id}"
    data_dir = f"{WORKSPACE}/muon-data/data"
    local_dir = args.output_dir.expanduser().resolve() / run_id
    local_dir.mkdir(parents=True, exist_ok=False)
    start_attempted = False
    output_ready = False
    try:
        print(f"Starting pod {pod_id}; local results: {local_dir}", flush=True)
        start_attempted = True
        runpod.resume_pod(pod_id)
        host, port, user = wait_for_ssh(runpod, pod_id, args.startup_timeout)
        ssh = ["ssh"] + ssh_options(key, port) + [f"{user}@{host}"]
        print("Verify the SSH host fingerprint if prompted.", flush=True)
        remote(ssh, "true")
        remote(ssh, f"mkdir -p {shlex.quote(run_dir)} {shlex.quote(data_dir)}")
        scp(key, host, port, user, TRAINER, run_dir + "/train_gpt2.py")
        scp(key, host, port, user, DOWNLOADER, data_dir + "/cached_fineweb10B.py")
        gpu_count = 8 if args.full_run else 1
        remote(ssh, " && ".join([
            "python -c " + shlex.quote(
                "import numpy, torch; "
                f"count = torch.cuda.device_count(); "
                f"assert count >= {gpu_count}, f'Expected at least {gpu_count} CUDA GPUs, found {{count}}'; "
                "print('GPUs:', count, [torch.cuda.get_device_name(i) for i in range(count)], flush=True)"
            ),
            f"python -m venv {shlex.quote(run_dir + '/downloader-venv')}",
            f"{shlex.quote(run_dir + '/downloader-venv/bin/python')} -m pip install 'huggingface_hub>=0.25,<2'",
        ]))
        remote(ssh, f"{shlex.quote(run_dir + '/downloader-venv/bin/python')} "
               f"{shlex.quote(data_dir + '/cached_fineweb10B.py')} {103 if args.full_run else 1}")
        remote(ssh, f"ln -s {shlex.quote(data_dir)} {shlex.quote(run_dir + '/data')}")
        remote(ssh, f"mkdir -p {shlex.quote(run_dir + '/logs')} && touch {shlex.quote(run_dir + '/training.log')}")
        output_ready = True
        run_name = "eight-GPU run" if args.full_run else "one-GPU run"
        print(f"Running {run_name}. Results will be copied to: {local_dir}", flush=True)
        command = (
            f"cd {shlex.quote(run_dir)} && set -o pipefail && "
            f"python -u -m torch.distributed.run --standalone --nproc_per_node={gpu_count} "
            "train_gpt2.py 2>&1 | tee training.log"
        )
        remote(ssh, "bash -c " + shlex.quote(command))
        print("Training completed successfully.", flush=True)
    finally:
        if start_attempted:
            original_error = sys.exc_info()[1]
            copy_error = None
            try:
                if output_ready:
                    print(f"Copying results to {local_dir} before stopping the pod.", flush=True)
                    try:
                        copy_results(key, host, port, user, run_dir, local_dir)
                    except (OSError, subprocess.SubprocessError) as exc:
                        copy_error = exc
                        print(f"Result copy failed: {exc}. Files left on container disk will be lost.",
                              file=sys.stderr)
                    else:
                        print(f"Results copied to {local_dir}", flush=True)
            finally:
                print(f"Stopping pod {pod_id}; verify its status in the Runpod console.", flush=True)
                try:
                    stop_pod(runpod, pod_id)
                except Exception:
                    if original_error is not None:
                        print(f"Original failure: {original_error}", file=sys.stderr)
                    print("Pod cleanup failed; stop it manually in the Runpod console.", file=sys.stderr)
                    raise
                print("Pod stopped. Any persistent storage may still incur charges.", flush=True)
            if copy_error is not None and original_error is None:
                raise RuntimeError("Training finished but copying results to this Mac failed") from copy_error


def main(argv=None):
    print("WARNING: runpod_launch.py is deprecated and may be removed.", file=sys.stderr)
    args = parse_args(argv)
    try:
        import runpod
        import runpod.error
        from dotenv import load_dotenv
    except ImportError:
        print(f"Install launcher dependencies: python -m pip install -r {HERE / 'requirements-runpod.txt'}",
              file=sys.stderr)
        return 1
    load_dotenv(dotenv_path=ENV_FILE, override=False)
    runpod.api_key = os.environ.get("RUNPOD_API_KEY")
    try:
        launch(args, runpod)
    except KeyboardInterrupt:
        print("Interrupted. Verify the pod is stopped in the Runpod console.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, subprocess.SubprocessError, runpod.error.RunPodError) as exc:
        print(f"Launch failed: {exc}. Verify the pod is stopped in the Runpod console.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())