#!/usr/bin/env bash
# Run on the Mac: bash launch_tmux.sh <pod-ip> <ssh-port> <pod-id> [training-script.py] [--download_shards=N]
# The training script defaults to train_gpt2.py next to this launcher.
# The pod must already be running, with tmux and CUDA-enabled PyTorch installed.
# A reusable /workspace/muon-venv is created if needed.
# Training and its logs use the pod's persistent /workspace network volume.
# --download_shards=N downloads N FineWeb10B training shards and the validation shard before training.
# The tmux session name uses the training script's filename and a unique run ID.
set -euo pipefail

if [[ ${1:-} == "--pod-setup" ]]; then
    venv_dir=${2:?Missing virtual environment path}
    python="$venv_dir/bin/python"
    if [[ ! -e "$venv_dir" && ! -L "$venv_dir" ]]; then
        echo "Creating $venv_dir with access to the pod's Python packages."
        python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available"'
        python3 -m venv --system-site-packages "$venv_dir"
    elif [[ ! -x "$python" ]]; then
        echo "Existing environment $venv_dir has no working Python; inspect it before retrying." >&2
        exit 1
    fi

    "$python" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available"'
    download_shards=${3:-0}
    missing=$("$python" -c 'import importlib.util, sys; names = ["numpy", "runpod"]; names += ["huggingface_hub"] if sys.argv[1] != "0" else []; print(" ".join(name for name in names if importlib.util.find_spec(name) is None))' "$download_shards")
    packages=()
    if [[ " $missing " == *" numpy "* ]]; then
        packages+=(numpy)
    fi
    if [[ " $missing " == *" runpod "* ]]; then
        packages+=('runpod>=2.0,<3')
    fi
    if [[ " $missing " == *" huggingface_hub "* ]]; then
        packages+=(huggingface_hub)
    fi
    if (( ${#packages[@]} )); then
        echo "Installing missing Python dependencies: ${packages[*]}"
        "$python" -m pip install "${packages[@]}"
    fi
    "$python" -c 'import numpy, runpod, torch; assert torch.cuda.is_available(), "CUDA is not available"'
    if (( download_shards > 0 )); then
        "$python" -c 'import huggingface_hub'
    fi
    echo "Pod Python prerequisites are ready."
    exit 0
fi

if [[ ${1:-} == "--pod-session" ]]; then
    run_dir=$2
    pod_id=$3
    download_shards=${4:-0}
    cd "$run_dir"

    if [[ -n ${RUNPOD_POD_ID:-} && "$RUNPOD_POD_ID" != "$pod_id" ]]; then
        echo "Pod ID does not match the current pod; training was not started." >&2
        exit 1
    fi
    check_shards() {
        shopt -s nullglob
        train_shards=(data/fineweb10B/fineweb_train_*.bin)
        val_shards=(data/fineweb10B/fineweb_val_*.bin)
        echo "Found ${#train_shards[@]} training shards and ${#val_shards[@]} validation shards."
        if (( ${#train_shards[@]} < 33 || ${#val_shards[@]} == 0 )); then
            echo "The full run requires at least 33 training shards and one validation shard. Stop the pod manually." >&2
            return 1
        fi
    }
    if (( download_shards == 0 )); then
        check_shards
    fi
    /workspace/muon-venv/bin/python -c 'import numpy, runpod, torch; assert torch.cuda.is_available(), "CUDA is not available"'
    read -rs -p "RunPod API key: " RUNPOD_API_KEY
    echo
    if [[ -z "$RUNPOD_API_KEY" ]]; then
        echo "No API key entered; training was not started. Stop the pod manually." >&2
        exit 1
    fi
    if (( download_shards > 0 )); then
        echo "Downloading $download_shards training shards and the validation shard. Detach with Ctrl+B, then D."
        if ! /workspace/muon-venv/bin/python -u downloader/cached_fineweb10B.py "$download_shards"; then
            echo "Dataset download failed; training was not started. Stop the pod manually." >&2
            exit 1
        fi
        check_shards
    fi
    export STOP_RUNPOD_ON_EXIT=1 RUNPOD_POD_ID="$pod_id" RUNPOD_API_KEY
    echo "Starting training in $run_dir. Detach with Ctrl+B, then D."
    echo "Pod shutdown is best effort; verify its status in the RunPod dashboard."
    if /workspace/muon-venv/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=1 \
        train_gpt2.py 2>&1 | tee training.log; then
        echo "Training process exited successfully."
    else
        status=$?
        echo "Training failed with exit status $status; verify that the pod stops." >&2
        unset RUNPOD_API_KEY
        exit "$status"
    fi
    unset RUNPOD_API_KEY
    exit 0
fi

if (( $# < 3 )); then
    echo "Usage: bash $0 <pod-ip> <ssh-port> <pod-id> [training-script.py] [--download_shards=N]" >&2
    exit 2
fi
host=$1
port=$2
pod_id=$3
shift 3
trainer=""
download_shards=0
download_requested=false
for arg in "$@"; do
    case "$arg" in
        --download_shards=*)
            if [[ "$download_requested" == true ]]; then
                echo "Specify --download_shards only once." >&2
                exit 2
            fi
            download_requested=true
            count=${arg#--download_shards=}
            if [[ ! "$count" =~ ^[1-9][0-9]{0,2}$ ]] || (( count > 103 )); then
                echo "--download_shards must be an integer from 1 to 103." >&2
                exit 2
            fi
            download_shards=$count
            ;;
        -*)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
        *)
            if [[ -n "$trainer" ]]; then
                echo "Specify at most one training script." >&2
                exit 2
            fi
            trainer=$arg
            ;;
    esac
done
if [[ ! "$host" =~ ^[0-9.]+$ || ! "$port" =~ ^[0-9]+$ ||
      ! "$pod_id" =~ ^[a-zA-Z0-9_-]+$ ]] || (( 10#$port < 1 || 10#$port > 65535 )); then
    echo "Invalid pod IP, SSH port, or pod ID." >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
trainer="${trainer:-$script_dir/train_gpt2.py}"
downloader="$script_dir/data/cached_fineweb10B.py"
key="$HOME/.ssh/id_ed25519"
[[ -f "$trainer" && -f "$key" ]] || {
    echo "Missing trainer or SSH key: $trainer / $key" >&2
    exit 1
}
if (( download_shards > 0 )) && [[ ! -f "$downloader" ]]; then
    echo "Missing dataset downloader: $downloader" >&2
    exit 1
fi
run_id="$(date -u +%Y%m%d-%H%M%S)-$$-$RANDOM"
run_dir="/workspace/muon-runs/$run_id"
trainer_name=${trainer##*/}
trainer_name=${trainer_name%.py}
trainer_name=$(printf '%s' "$trainer_name" | LC_ALL=C tr -c 'A-Za-z0-9_-' '_')
session="${trainer_name:-trainer}-$run_id"
ssh_options=(-i "$key" -o IdentitiesOnly=yes -o StrictHostKeyChecking=ask -p "$port")
scp_options=(-i "$key" -o IdentitiesOnly=yes -o StrictHostKeyChecking=ask -P "$port")

echo "Preparing $run_dir on the running pod."
echo "Uploading training script: $trainer"
if ! ssh "${ssh_options[@]}" "root@$host" \
    "bash -s -- --pod-setup /workspace/muon-venv $download_shards" < "$script_dir/launch_tmux.sh"; then
    echo "Pod setup failed. Check the error above; the pod is still running." >&2
    exit 1
fi
if (( download_shards > 0 )); then
    ssh "${ssh_options[@]}" "root@$host" \
        "command -v tmux >/dev/null && mkdir -p /workspace/muon-data/data/fineweb10B /workspace/muon-runs && mkdir '$run_dir' && ln -s /workspace/muon-data/data '$run_dir/data' && mkdir '$run_dir/downloader' && ln -s /workspace/muon-data/data/fineweb10B '$run_dir/downloader/fineweb10B'"
    scp "${scp_options[@]}" "$downloader" "root@$host:$run_dir/downloader/cached_fineweb10B.py"
else
    ssh "${ssh_options[@]}" "root@$host" \
        "test -d /workspace/muon-data/data/fineweb10B && command -v tmux >/dev/null && mkdir -p /workspace/muon-runs && mkdir '$run_dir' && ln -s /workspace/muon-data/data '$run_dir/data'"
fi
scp "${scp_options[@]}" "$trainer" "root@$host:$run_dir/train_gpt2.py"
scp "${scp_options[@]}" "$script_dir/launch_tmux.sh" "root@$host:$run_dir/launch_tmux.sh"
echo "Connecting to tmux session $session."
echo "Enter your RunPod API key at the prompt; it will not be displayed."
echo "Reconnect later with: ssh -t -i \"$key\" -p $port root@$host 'tmux attach -t $session'"
ssh -tt "${ssh_options[@]}" "root@$host" \
    "tmux new-session -s '$session' 'bash $run_dir/launch_tmux.sh --pod-session $run_dir $pod_id $download_shards'"
