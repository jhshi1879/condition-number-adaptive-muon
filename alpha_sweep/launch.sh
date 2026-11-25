#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
script="$root/alpha_sweep/launch.sh"
python=${PYTHON:-/workspace/venv/bin/python}
mode=launch
if [[ ${1:-} == --run ]]; then
    mode=run
    shift
fi
if (( $# < 5 || $# > 7 )); then
    echo "Usage (on pod): bash alpha_sweep/launch.sh SWEEP_DIR POD_ID DEPLOY_KEY KNOWN_HOSTS DEPLOY_KEY_ID [OWNER/REPO] [RESULTS_PATH]" >&2
    exit 2
fi
sweep_dir=$1
pod_id=$2
key=$3
known_hosts=$4
key_id=$5
repo=${6:-jhshi1879/condition-number-adaptive-muon}
results=${7:-alpha_sweep/results/three-rounds}
if [[ $sweep_dir != /* || -e $sweep_dir || -e "$sweep_dir.console.log" ]]; then
    echo "Use a new absolute sweep directory; existing results/logs are never overwritten." >&2
    exit 1
fi
if [[ -n ${RUNPOD_POD_ID:-} && $RUNPOD_POD_ID != "$pod_id" ]]; then
    echo "Pod ID does not match this pod; nothing was started." >&2
    exit 1
fi
cd "$root"

if [[ $mode == launch ]]; then
    for session in alpha-sweep alpha-finalize alpha-post-analysis; do
        if tmux has-session -t "=$session" 2>/dev/null; then
            echo "Session $session already exists; inspect it before launching another sweep." >&2
            exit 1
        fi
    done
    printf -v command '%q ' bash "$script" --run "$sweep_dir" "$pod_id" "$key" "$known_hosts" "$key_id" "$repo" "$results"
    printf -v log '%q' "$sweep_dir.finalization.log"
    mkdir -p -- "$(dirname -- "$sweep_dir")"
    tmux new-session -d -s alpha-finalize -c "$root" \
        "tmux set-window-option remain-on-exit on && exec $command >> $log 2>&1"
    echo "Started alpha-finalize; it starts alpha-sweep and owns analysis, publication and shutdown."
    echo "Attach: tmux attach -t alpha-finalize"
    echo "Log: $sweep_dir.finalization.log"
    exit 0
fi

if [[ -z ${TMUX:-} ]]; then
    echo "The worker must run inside tmux; use the public launcher without --run." >&2
    exit 1
fi
printf -v training '%q ' "$python" -u "$root/alpha_sweep/train_gpt2_cond_muon.py" \
    --sweep --alpha-grid 0 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 \
    --screen-steps 100 --refine-steps 100 --finalist-steps 500 \
    --finalist-seeds 0 1 --top-k 2 --seed 0 --val-tokens 1048576 \
    --val-loss-every 25 --device-batch-size 8 --save-every 0 \
    --warmdown-iters 10 --output-dir "$sweep_dir"
printf -v log '%q' "$sweep_dir.console.log"
tmux new-session -d -s alpha-sweep -c "$root" \
    "tmux set-window-option remain-on-exit on && exec $training > $log 2>&1"
for ((waited=0; waited<120; waited++)); do
    [[ -f "$sweep_dir/summary.json" ]] && break
    if [[ $(tmux list-panes -s -t =alpha-sweep -F '#{pane_dead}') == 1 ]]; then
        echo "Training exited before initializing its summary. Inspect $sweep_dir.console.log; pod not stopped." >&2
        exit 1
    fi
    sleep 1
done
if [[ ! -f "$sweep_dir/summary.json" ]]; then
    echo "Training summary did not appear within 120s; finalization was not armed. Inspect both sessions." >&2
    exit 1
fi
state_dir="$sweep_dir/finalization"
"$python" "$root/alpha_sweep/finish_sweep.py" --on-pod --prepare --skip-preflight \
    --state-dir "$state_dir" --project-root "$root" --sweep-dir "$sweep_dir" \
    --repo "$repo" --pod-id "$pod_id" --ssh-key "$key" --known-hosts "$known_hosts" \
    --deploy-key-id "$key_id" --results-relative "$results"
exec "$python" -u "$state_dir/sources/alpha_sweep/finish_sweep.py" \
    --on-pod --state-dir "$state_dir"
