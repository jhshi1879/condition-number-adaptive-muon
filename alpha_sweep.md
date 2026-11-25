---
name: alpha-sweep
description: Run the three-round fixed-Muon-alpha experiment on a Runpod GPU, analyze it, publish checkpoint-free results, and stop compute entirely from tmux.
---

# Three-round Muon alpha sweep

Use this repository-specific skill when asked to reproduce or extend the fixed-alpha
experiment. This is the skill source document; to register it with a client that
requires the standard layout, copy it to that client's `alpha-sweep/SKILL.md`.
Do not launch a new pod or rerun training merely to read this skill.

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
| Round 1 | Alphas 0 to 3 inclusive in increments of 0.25; 100 steps; seed 0 |
| Round 2 | Two best regions, radius 0.25, spacing 0.0625; 100 steps; seed 0 |
| Round 3 | Two best round-2 regions, radius 0.0625, spacing 0.03125; 500 steps per seed, seeds 0 and 1 |
| Objective | Lowest mean **final** round-3 validation loss over both seeds |
| Learning rates | Muon matrices 0.00036; AdamW head 0.0036 |

Merge overlapping candidates; clip to the original 0-3 search bounds. Every
trial starts from scratch. This is a fixed-step experiment, **not one epoch**:
100 steps consume 52,428,800 tokens and 500 consume 262,144,000. The upper bound
with two disjoint interior regions is 13,100 optimizer steps; actual counts vary.
The historical run completed 13 + 13 + 14 = 40 trials (9,600 optimizer steps).

Fixed alpha uses `(X X.T + 0.01^2 I)^(-alpha/2) X` on Frobenius-normalized
momentum, transposing tall matrices and preserving output normalization.
Use Gram eigendecomposition, not the degree-32 polynomial for inverse powers.
Alpha is nonnegative and may exceed 1. Omitting fixed alpha preserves adaptive
behavior. Divergent trials are recorded and excluded; unexpected child errors
abort the sweep. Seeds do not promise bitwise-deterministic CUDA execution.

## Prerequisites (prepare once; no smoke run)

Use a running Linux GPU pod with Bash, tmux, Git, OpenSSH, Python with a
CUDA-enabled PyTorch 2.x, NumPy, and Matplotlib. The historical runtime used
Python 3.12 and PyTorch 2.8/cu128. Keep the environment, repository, data, logs,
and checkpoints under the durable mount, e.g. `/workspace`.

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
  alpha_sweep/results/new-run
```

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

After **all three rounds** complete, `post_analysis.py` produces:

- Per-round final validation loss versus alpha.
- Validation loss versus training step for round winners.
- Runtime versus alpha, seed variability, and search coverage.
- `analysis/analysis.ipynb` with five rendered figures and reproducible code.
- Exactly two paragraphs in `analysis/experiment_report.md`, also in the notebook.
- `analysis/analysis_manifest.json` with source-summary and output hashes.

### Required notebook writeup

Use **Round 1**, **Round 2**, and **Round 3** in reader-facing prose, figure
titles, legends, and tables, rather than descriptive stage names. Internal
result-schema keys remain unchanged for compatibility. Use `$...$` for inline
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
