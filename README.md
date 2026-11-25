# Condition-number-adaptive Muon

Experiment scripts based on the [modded-nanogpt Muon record](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2024-10-10_Muon). The original code is MIT-licensed; see [LICENSE](LICENSE).

`train_gpt2_cond_muon.py` implements the condition-number-adaptive optimizer; `train_gpt2.py` and `train_gpt2_adamw.py` are comparison trainers. See [study_notes.md](study_notes.md) for the algorithm and limitations.

## Reusable alpha-sweep workflow

See [alpha_sweep.md](alpha_sweep.md) for the experiment skill, historical results,
credentials, monitoring, recovery, and safe publication/shutdown.
[alpha_sweep/launch.sh](alpha_sweep/launch.sh) starts the three-round sweep and
finalizer in pod-side tmux, skipping standalone preflight and smoke runs while
retaining runtime validation and publication verification.

## Training

Use Python 3.10 or later, NumPy, and a CUDA-enabled PyTorch 2.x installation with `torch.compile` support. Install PyTorch using the instructions for your CUDA environment. The trainers require NVIDIA GPUs, NCCL, and `torchrun`; they do not run on CPU or macOS GPU.

To download FineWeb10B data, install `huggingface_hub`, then run:

```sh
python data/cached_fineweb10B.py 33
torchrun --standalone --nproc_per_node=1 train_gpt2_cond_muon.py
```

The downloader saves shards in `data/fineweb10B/`. The full dataset has 103 training shards; supply that number to download them all. Training defaults are configured in each trainer. Run `python train_gpt2_cond_muon.py --help` to see the available overrides. Logs and checkpoints are saved under `logs/`.

For a running RunPod pod, `launch_tmux.sh` uploads a trainer and optionally the downloader. It requires SSH access, tmux, CUDA-enabled PyTorch, and a persistent `/workspace` volume. See its header for usage. The optional shutdown feature requires a RunPod API key and the dependencies in `requirements-runpod.txt`. `runpod_launch.py` is deprecated.

## Analysis and tests

The plots and notebooks reference historical training logs. The AdamW and conditional-Muon logs under `runpod-results/` are not included; restore them at the paths specified in `plots/loss_vs_time.py` (or update the paths) before rerunning the analysis. Matplotlib and a Jupyter environment are needed for plots and notebooks.

Run the local tests with:

```sh
python -m unittest discover -s tests -v
```

The distributed optimizer tests use CPU and Gloo and do not validate CUDA training or RunPod execution.
