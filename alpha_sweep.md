---
name: alpha-sweep
description: Run a configurable multi-round fixed-Muon-alpha experiment on a Runpod GPU, analyze it, publish checkpoint-free results, and stop compute entirely from tmux.
---

# Multi-round Muon alpha sweep

Use this repository-specific skill when asked to reproduce or extend the fixed-alpha
experiment. This is the skill source document; to register it with a client that
requires the standard layout, copy it to that client's `alpha-sweep/SKILL.md`.
Do not launch a new pod or rerun training merely to read this skill.

## 1. Set up a low-cost Runpod GPU and durable volume

When the user authorizes a new experiment, begin by provisioning **one GPU pod
for less than $1/hour in total pod compute charges**. This ceiling is strict:
do not select a $1/hour-or-higher pod, or interpret it as permission for a
more expensive fallback. Storage bills separately. This section describes the
setup workflow; reading the skill alone does not authorize provisioning.

1. Follow the installed Runpod routing, pod-deploy, and network-volume
   procedures and the training-pod golden path. Check the authenticated account,
   available balance, registered SSH public key, and existing volumes before
   creating resources. Read current GPU **pod** pricing, stock by data center,
   CUDA availability, network-storage availability and storage pricing. Choose
   the least expensive available CUDA-compatible GPU that fits the workload,
   rather than assuming historical prices or stock still apply.
2. The historical 29.9M-parameter workload used approximately 6 GiB of GPU
   memory during a live trial on a 24 GB card; allow headroom for compilation,
   allocator use, and variations. Prefer a card with at least 16 GB of VRAM;
   this is a conservative selection, not a proven minimum. Match a current
   official PyTorch image to its supported CUDA version and pin the minimum
   host CUDA version to the image's version (e.g. 12.8 for a CUDA 12.8 image).
3. Before creating billable resources, state the chosen GPU and **actual hourly
   pod rate**, the volume size and storage billing, and an estimated duration
   and total cost if a credible estimate exists. Ensure available credit covers
   the planned run and storage; runtime is variable. If no suitable GPU below
   $1/hour is in stock, report that and ask whether to wait or change the
   budget—never create an over-budget pod.
4. Choose an available data center with network storage and create or reuse an
   appropriately sized **network volume in that same data center**. For the
   checkpoint-free experiment, **20 GB** is a starting estimate: the 33
   training shards plus validation shard occupy approximately 6.8 GB, with
   additional room for the environment, compilation caches, logs and notebook.
   Check actual free space before launching. Size larger for additional datasets
   or existing data; do not delete existing volumes to
   make room without authorization. State that volume charges continue until
   deletion. Network volumes require a Secure Cloud pod in the volume's data
   center and an explicit `/workspace` mount.
5. Create the pod with the chosen GPU and current official PyTorch image,
   CUDA floor, co-located `/workspace` mount, registered SSH key, and direct
   SSH (`22/tcp`, with SSH enabled). Expose no web ports unless requested.
   Read the created pod back, confirm its GPU and volume mount, and obtain its
   **current** direct SSH host and port. Verify the SSH host key before copying
   files or connecting; do not disable host-key checking. Never reuse an old
   pod's IP or port after a restart.
6. Put this repository, Python environment, FineWeb data, compilation caches,
   and outputs on `/workspace`, not the ephemeral container disk. Use the
   prerequisite instructions below to fetch the 33 shards if absent and
   prepare a repository-only deploy key and pod-scoped shutdown access. The
   pod's `RUNPOD_API_KEY` must have permission to read and stop that pod; do
   not assume it exists or copy an account-wide GitHub credential to the pod.
   Launch the **real** configured run directly in tmux, as described below.
   Do not add a separate preflight or smoke training run.

If creation fails because stock changed, re-read availability. Keep any volume
already created; retry once in its data center if stock remains, otherwise ask
before waiting, choosing another eligible GPU, or moving data centers. Do not
silently create another volume or exceed the price ceiling. After completion,
verify the pod is stopped independently and retain the volume until the user
authorizes its deletion.

## Execution policy

- **Skip separate preflight and smoke tests.** Start the real experiment directly:
  no one-step GPU trial, mini-sweep, benchmark, or Git push dry-run. The launcher
  passes `--skip-preflight` during finalizer preparation. Preparation still freezes
  source files; it is not a training run.
- Do not skip credential permissions, pinned SSH host keys, input validation,
  completed-round checks, output checksums, secret scanning, fresh publication
  verification, or stop authorization. The live finalizer authenticates at runtime.
- All training, plotting, publication, and shutdown run on the pod in tmux.
  Nothing depends on the Mac staying awake or connected.
- Provision only with user authorization and a live price/stock read. Prefer
  one inexpensive CUDA-compatible GPU, a persistent volume, and adequate credit.
  Never blindly reuse historical pod addresses, prices, or credentials.
- Stop, do not terminate. Stopping releases compute and ends tmux; it clears
  container disk but preserves retained volumes, which continue billing.

## Scientific setup

The trainer is `alpha_sweep/train_gpt2_cond_muon.py`; shared model definitions
are in `alpha_sweep/models.py`. Use Python to launch its sweep controller, not
`torchrun` (the controller launches each distributed child itself).

| Setting | Value |
| --- | --- |
| Model | 6 layers, 6 heads, width 384; 29,915,520 parameters; tied embeddings/output |
| Dataset | FineWeb10B; historical download: 33 x 100M training tokens, 100M validation tokens |
| Global batch | 512 sequences x 1,024 tokens = 524,288 tokens/step |
| Device batch | 8, with gradient accumulation |
| Validation | Same first 1,048,576 held-out tokens every 25 steps and at the end |
| Rounds | Configurable; 100 steps and seed 0 in each non-final round; 500 steps and seeds 0 and 1 in the final round |
| Search | Round 1 covers the requested initial grid; later rounds refine the two best regions and stay within the original bounds |
| Objective | Lowest mean **final** validation loss over the final-round seeds |
| Learning rates | Muon matrices 0.00036; AdamW head 0.0036 |

Merge overlapping candidates; clip to the configured search bounds. Every
trial starts from scratch. This is a fixed-step experiment, **not one epoch**:
100 steps consume 52,428,800 tokens and 500 consume 262,144,000. Estimate the
maximum number of trials and total cost from the configured grid and rounds;
overlap, boundary clipping and divergence affect actual counts. The historical
three-round run completed 13 + 13 + 14 = 40 trials (9,600 optimizer steps).
For the five-round 0-10 grid with spacing 0.25, a disjoint interior upper bound
is 17,900 steps (41 + 18 + 10 + 10 short trials and 10 final candidates with
two seeds). Disable all checkpoints for the new sweep.

Fixed alpha uses `(X X.T + 0.01^2 I)^(-alpha/2) X` on Frobenius-normalized
momentum, transposing tall matrices and preserving output normalization.
Use Gram eigendecomposition, not the degree-32 polynomial for inverse powers.
Alpha is nonnegative and may exceed 1. Omitting fixed alpha preserves adaptive
behavior. Divergent trials are recorded and excluded; unexpected child errors
abort the sweep. Seeds do not promise bitwise-deterministic CUDA execution.

## Prerequisites (prepare once; no smoke run)

Use a running Linux GPU pod with Bash, tmux, Git, OpenSSH, Python with a
CUDA-enabled PyTorch 2.x, NumPy, and Matplotlib. The historical runtime used
Python 3.12 and PyTorch 2.8/cu128. Keep the environment, repository, data and
logs under the durable mount, e.g. `/workspace`.

The default interpreter is `/workspace/venv/bin/python`; override with `PYTHON`.
Install dependencies only if absent. For a fresh dataset, install
`huggingface_hub` and run `python data/cached_fineweb10B.py 33` once from the
repository root. Do not redownload existing shards. A new sweep requires an
unused absolute results directory; checkpoints cannot be resumed.

For autonomous GitHub publication:

1. With the user's permission, generate an owner-only Ed25519 deploy key in a
   private directory under **`/dev/shm`**, not on the persistent volume:
   `mkdir -m 700 /dev/shm/alpha-publisher`, then
   `ssh-keygen -q -t ed25519 -N '' -f /dev/shm/alpha-publisher/id_ed25519`.
2. Register only its public key as a **write-enabled deploy key** on the exact
   destination repository. Use the existing GitHub-authenticated machine's
   `gh api repos/OWNER/REPO/keys` and record the returned key ID. Never copy an
   account-wide GitHub token onto the pod.
3. Pin GitHub's SSH host keys from authenticated HTTPS metadata:
   `gh api meta --jq '.ssh_keys[] | "github.com " + .'`.
   Transfer that public output into `/dev/shm/alpha-publisher/known_hosts`.
   Do not rely on an unverified `ssh-keyscan` result.
4. The pod runtime must expose `RUNPOD_API_KEY` (or its original process
   environment must contain it) with permission to read and stop **this pod**.
   Never paste keys into chat, commands, logs, notebooks, or Git.

Keep the RAM-only private key until publication succeeds. The finalizer destroys
it before requesting stop. Revoke the public deploy-key entry afterward from
the GitHub-authenticated machine using the recorded ID.

## Launch directly in tmux

Run on the pod, from the repository root, substituting the current pod ID and
deploy-key ID. The last two arguments optionally select repository and output
location in GitHub:

```sh
bash alpha_sweep/launch.sh \
  /workspace/alpha-sweep-new-run \
  POD_ID \
  /dev/shm/alpha-publisher/id_ed25519 \
  /dev/shm/alpha-publisher/known_hosts \
  DEPLOY_KEY_ID \
  jhshi1879/condition-number-adaptive-muon \
  alpha_sweep/results/new-run \
  --rounds 5 \
  --alpha-grid 0 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 \
    3 3.25 3.5 3.75 4 4.25 4.5 4.75 5 5.25 5.5 5.75 6 \
    6.25 6.5 6.75 7 7.25 7.5 7.75 8 8.25 8.5 8.75 9 9.25 9.5 9.75 10
```

Choose the positive round count and initial grid for each experiment. The
launcher uses 100 steps in each non-final round and 500 steps per seed in the
final round; it disables periodic and final checkpoints. With one round, the
initial grid is evaluated at the final budget.

`alpha-finalize` starts `alpha-sweep`, waits for the real controller to create
its summary, freezes an explicit source allowlist without preflight probes, and
runs the finalizer. It never starts a smoke trial or provisions compute.
Both sessions retain exited panes for diagnosis. Existing session names and
output paths are refused rather than overwritten. Avoid concurrent sweeps on
the same pod. After launching, verify the live sessions and summary; a command
return alone is not proof of readiness.

```sh
tmux list-panes -a -F '#{session_name} dead=#{pane_dead}'
tmux attach -t alpha-sweep
tmux attach -t alpha-finalize
tail -n 10 /workspace/alpha-sweep-new-run.console.log
tail -n 10 /workspace/alpha-sweep-new-run.finalization.log
```

Detach with Ctrl-b, then d. The controller's `summary.json` records actual
grids, completed trials, and the active trial. Each trial has `console.log`
and full-precision `result.json`. Training logs print only four decimal places;
the logged train loss is the final microbatch loss, not a batch mean.

## Analysis, publication, shutdown

After **all configured rounds** complete, `post_analysis.py` produces:

- Per-round final validation loss versus alpha.
- Validation loss versus training step for round winners.
- Runtime versus alpha, seed variability, and search coverage.
- `analysis/analysis.ipynb` with five rendered figures and reproducible code.
- Exactly two paragraphs in `analysis/experiment_report.md`, also in the notebook.
- `analysis/analysis_manifest.json` with source-summary and output hashes.

### Required notebook writeup

Use numbered labels such as **Round 1** and **Round 2** in reader-facing prose,
figure titles, legends, and tables, rather than descriptive stage names. Use `$...$` for inline
LaTeX and `$$...$$` for display equations in notebook Markdown.

Include two additional short Markdown paragraphs **in the notebook itself**,
separate from the two-paragraph results report. Place them before the plots.
These are required for every future generated notebook, not just this historical
experiment. Keep the standalone experiment report at exactly two paragraphs.

1. **Model and experiment setup (reproducibility).** State the architecture,
   parameter count, tied weights, tokenizer/vocabulary and sequence length;
   dataset source/version, shard identifiers or checksums, training subset and
   exact held-out selection; optimizer assignments, momentum, learning rates,
   schedule/warmdown, precision, global/device batches and accumulation; actual
   alpha grids, per-round step/token budgets, initialization/data-order seeds,
   validation cadence/budget and final-loss aggregation criterion. Include the
   exact launch command, source commit or source-file hashes, GPU, and
   Python/PyTorch/CUDA versions (plus relevant dependency versions). Distinguish
   recorded settings from inferred defaults; explicitly identify any missing
   provenance instead of claiming full reproducibility. Mention CUDA
   nondeterminism and that each trial starts from scratch. Use the actual run's
   configuration, not blindly copied historical values.

2. **Theory and formalism (especially alpha).** Define gradient $G_t$, EMA
   momentum $M_t=\beta M_{t-1}+(1-\beta)G_t$, and normalized matrix
   $X=M_t/\max(\lVert M_t\rVert_F,\epsilon)$, with the tall-matrix transpose
   convention. Explain the fixed-alpha transform and final update:
   $Y_\alpha=(XX^\top+\delta^2 I)^{-\alpha/2}X$,
   $\Delta_\alpha=\sqrt{r}\,Y_\alpha/\max(\lVert Y_\alpha\rVert_F,\epsilon)$,
   and $W_{t+1}=W_t-\eta\Delta_\alpha$, where $\delta=0.01$ and
   $r=\min(m,n)$ for each transformed matrix block (transpose back when needed).
   For $X=U\operatorname{diag}(\sigma_i)V^\top$, show explicitly that
   $\sigma_i\mapsto\sigma_i(\sigma_i^2+\delta^2)^{-\alpha/2}$
   before output normalization. Explain that alpha controls spectral flattening,
   not the learning rate: alpha 0 preserves the momentum direction, alpha 1
   approximates polar normalization for singular values well above the floor,
   and alpha above 1 reverses the relative weighting of those above-floor
   singular values and can amplify noise. In the **unregularized positive-spectrum
   idealization**, the map is $\sigma_i^{1-\alpha}$ and
   $\kappa(Y_\alpha)=\kappa(X)^{|1-\alpha|}$; do not present this as exact for the
   implemented regularized transform or zero singular values. Explain the
   floor's role near zero (zero singular values remain zero), Frobenius
   rescaling, and eigendecomposition. Distinguish the swept constant alpha from
   the original moment-adaptive alpha; do not claim the sweep optimizes that
   adaptive rule or proves a global optimum.

Required missing rows, missing winner curves, incomplete budgets, inconsistent
rankings, or mismatched losses are errors, never fabricated success.
Standalone analysis: `python alpha_sweep/post_analysis.py --sweep-dir RUN_DIR`.

The finalizer publishes **only its source allowlist, notebook, and report**:
never checkpoints, datasets, credentials, raw trial directories, or an entire
dirty worktree. It builds an isolated Git commit, pushes without force, fetches
again to verify hashes/ancestry, and records the commit URL in
`RUN_DIR/finalization/state.json`. It then destroys the private deploy key,
closes finished task sessions, and requests stop **inside tmux**.

Inspect every pane with `tmux list-panes -s -t =SESSION -F '#{pane_dead}'`
before closing a session. Do not use a session-only `display-message` target to
infer another pane's liveness: the first experiment's shutdown stalled on this
bug, which has now been fixed. Never kill a live training process to hide a failure.

Self-stop can end the finalizer before it observes `EXITED`. Verify pod status
with an independent Runpod read afterward; never claim stopped from a request
alone. Revoke the task deploy-key entry with
`gh api --method DELETE repos/OWNER/REPO/keys/KEY_ID`, then verify its absence.
Retain the volume unless the user explicitly authorizes deletion.

## Recovery

- SSH loss: reconnect and attach; never start a second sweep blindly.
- Training failure: inspect trial logs and `summary.json`. No automatic shutdown
  occurs; report that the pod may still bill. Fix the issue and use a new output
  directory for a fresh run, not a checkpoint-resume claim.
- Publication/network failure: inspect finalization state/logs. Transient errors
  retry; explicit auth or validation failures stop without shutting down.
- If publication succeeded but stop failed, preserve the recorded commit and
  resume the fixed finalizer from the same state inside `alpha-finalize`.
  It must not require the destroyed deploy key or republish results:
  `tmux respawn-pane -t alpha-finalize:0.0 'PYTHON -u /path/to/finish_sweep.py --on-pod --state-dir RUN_DIR/finalization >> RUN_DIR.finalization.log 2>&1'`.
  Substitute real paths and confirm the old pane is dead first.
- If RAM keys were lost before publication, obtain permission for a replacement
  repository-only key; never replace access controls with disabled host checks.

## Historical result and next experiment

On 2026-11-25 the 40-trial run selected **alpha 0.9375** with mean final
validation loss **5.153156**, population standard deviation **0.005932** over
seeds 0/1 (losses 5.147224 and 5.159088). Alpha 0.96875 was next at 5.172941;
alpha 1.0 reached 5.195202. The winner is the **lower edge of the confirmation
grid**, so this is not evidence of a global optimum. Longer training shifted
the promising region: a follow-up should compare lower alphas at the same
500-step budget and more seeds, not simply assume ever-finer short screening
guarantees the best long-budget alpha. Ask for authorization before new runs.

Published [notebook](alpha_sweep/results/three-rounds/analysis.ipynb) and
[two-paragraph report](alpha_sweep/results/three-rounds/experiment_report.md);
experiment commit `1307c00ade72a54bb706b259dd8fd55b0d5ffe33`.
Historical hardware was an RTX PRO 4000 Blackwell; its observed compute rate
was $0.57/hour, **not a current price quote**. Do not restart the stopped
historical pod just to inspect this workflow.
