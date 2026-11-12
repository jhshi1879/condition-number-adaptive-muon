import argparse
import math
import os
import sys

with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import atexit
import glob
import time
import uuid
from dataclasses import dataclass

import numpy as np
import torch
import torch._inductor.config as config
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def register_pod_shutdown():
    if os.environ.get("STOP_RUNPOD_ON_EXIT") == "1" and os.environ.get("RANK") == "0":
        def stop_runpod_on_exit():
            import runpod

            runpod.api_key = os.environ["RUNPOD_API_KEY"]
            pod_id = os.environ["RUNPOD_POD_ID"]
            print(f"Requesting stop for pod {pod_id}...", flush=True)
            runpod.stop_pod(pod_id)

        atexit.register(stop_runpod_on_exit)
register_pod_shutdown()

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

def polynomial_fractional_power(x, gram, exponent, degree=32, spectral_floor=0.01):
    """Approximate x (gram + spectral_floor**2 I)**((exponent - 1) / 2)."""
    nodes = torch.arange(degree + 1, device=x.device, dtype=x.dtype)
    theta = math.pi * (nodes + 0.5) / (degree + 1)
    samples = ((theta.cos() + 1) * 0.5 + spectral_floor**2).pow((exponent - 1) / 2)
    coefficients = (2.0 / (degree + 1)) * (
        torch.cos(nodes[:, None] * theta[None, :]) @ samples
    )
    coefficients[0] *= 0.5

    b1 = torch.zeros_like(x)
    b2 = torch.zeros_like(x)
    for k in range(degree, 0, -1):
        b0 = 2 * (2 * (gram @ b1) - b1) - b2 + coefficients[k] * x
        b2, b1 = b1, b0
    result = 2 * (gram @ b1) - b1 - b2 + coefficients[0] * x
    return result

def moment_fractional_power(matrix, kappa0, eps):
    x = matrix.float()
    x = x / x.norm().clamp_min(eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    gram = x @ x.T
    rank_bound = min(matrix.shape)
    m2 = gram.diagonal().sum() / rank_bound
    m4 = gram.square().sum() / rank_bound
    ratio = (m4 / m2.square().clamp_min(torch.finfo(m2.dtype).tiny)).clamp_min(1)
    log_kappa = 1.64485 * ratio.log().sqrt()
    log_kappa0 = math.log(kappa0)
    alpha = torch.where(
        log_kappa <= log_kappa0,
        torch.zeros_like(log_kappa),
        (1 - log_kappa0 / log_kappa.clamp_min(log_kappa0)).clamp(0, 1),
    )
    result = polynomial_fractional_power(x, gram, 1 - alpha)
    return (result.T if transposed else result), alpha

class ConditionalNumberAdaptiveMuon(torch.optim.Optimizer):
    """
    EMA momentum followed by a moment-adaptive fractional spectral transform.

    Spectral moments estimate a log-normal condition number each step. A degree-32
    Chebyshev polynomial approximates the fractional power with a 0.01 spectral
    floor. This approximation trades accuracy for speed, especially near zero.
    Momentum buffers from the earlier unnormalized, Nesterov version are not
    compatible with this optimizer's EMA momentum.

    Matrix work is distributed by parameter across ranks. Momentum and adaptive
    coefficients remain replicated so that rank-zero checkpoints contain full state.
    DDP must provide the same synchronized gradients on all ranks.

    The optimizer expects 2D matrices; embeddings and the output head remain on AdamW.
    """
    def __init__(
        self,
        params,
        lr=3e-4,
        momentum=0.95,
        kappa0=10.0,
        eps=1e-7,
    ):
        if kappa0 <= 1:
            raise ValueError("kappa0 must be greater than 1")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            kappa0=kappa0,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        params = [p for group in self.param_groups for p in group['params']]
        costs = []
        for p in params:
            rows, cols = p.shape
            cost = rows * cols * min(rows, cols)
            if rows == 3 * cols:
                cost = rows * cols * cols
            costs.append(cost)
        loads = [0] * world_size
        owners = {}
        for i in sorted(range(len(params)), key=lambda i: costs[i], reverse=True):
            owner = min(range(world_size), key=lambda r: loads[r])
            owners[params[i]] = owner
            loads[owner] += costs[i]

        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad, alpha=1 - group['momentum'])
                update = buf

                matrices = update.split(update.size(1)) if update.size(0) == 3 * update.size(1) else (update,)
                state['step'] = state.get('step', 0) + 1
                alphas = state.setdefault('alphas', [None] * len(matrices))
                if owners[p] != rank:
                    continue
                transformed = []
                for i, matrix in enumerate(matrices):
                    result, alphas[i] = moment_fractional_power(
                        matrix, group['kappa0'], group['eps']
                    )
                    rank_bound = min(matrix.shape)
                    result = result * (rank_bound**0.5 / result.norm().clamp_min(group['eps']))
                    transformed.append(result.to(matrix.dtype))

                update = torch.cat(transformed, dim=0) if len(transformed) > 1 else transformed[0]
                p.data.add_(update, alpha=-group['lr'])

        if world_size > 1:
            buckets = {}
            for p in params:
                key = (owners[p], p.device, p.dtype)
                buckets.setdefault(key, []).append(p)
            for (owner, device, dtype), bucket in buckets.items():
                alpha_count = sum(3 if p.size(0) == 3 * p.size(1) else 1 for p in bucket)
                if rank == owner:
                    packed = torch.cat([p.detach().reshape(-1) for p in bucket])
                    alpha_values = []
                    for p in bucket:
                        alphas = self.state[p].get('alphas')
                        count = 3 if p.size(0) == 3 * p.size(1) else 1
                        alpha_values.extend(alphas if alphas is not None else [None] * count)
                    packed_alphas = torch.stack([
                        alpha if alpha is not None else torch.zeros((), device=device, dtype=torch.float32)
                        for alpha in alpha_values
                    ])
                else:
                    packed = torch.empty(sum(p.numel() for p in bucket), device=device, dtype=dtype)
                    packed_alphas = torch.empty(alpha_count, device=device, dtype=torch.float32)
                dist.broadcast(packed, src=owner)
                dist.broadcast(packed_alphas, src=owner)
                if rank != owner:
                    offset = 0
                    alpha_offset = 0
                    for p in bucket:
                        p.copy_(packed[offset:offset + p.numel()].view_as(p))
                        offset += p.numel()
                        count = 3 if p.size(0) == 3 * p.size(1) else 1
                        alphas = self.state[p].get('alphas')
                        if alphas is not None:
                            for i in range(count):
                                alphas[i] = packed_alphas[alpha_offset + i].clone()
                        alpha_offset += count

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / (2 * config.n_layer)**0.5)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 6200 # number of iterations to run
    learning_rate : float = 0.0036
    warmup_iters : int = 0
    warmdown_iters : int = 10 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 1000 # every how many steps to save the checkpoint? 0 for only at the end
defaults = Hyperparameters()
parser = argparse.ArgumentParser(description="Train GPT-2 with Muon.")
parser.add_argument("--device-batch-size", type=int, default=defaults.device_batch_size)
parser.add_argument("--num-iterations", type=int, default=defaults.num_iterations)
parser.add_argument("--warmdown-iters", type=int, default=defaults.warmdown_iters)
parser.add_argument("--val-loss-every", type=int, default=defaults.val_loss_every)
parser.add_argument("--val-tokens", type=int, default=defaults.val_tokens)
options = parser.parse_args()
if options.device_batch_size <= 0 or options.num_iterations <= 0:
    parser.error("device batch size and number of iterations must be positive")
if not 0 < options.warmdown_iters <= options.num_iterations:
    parser.error("warmdown iterations must be positive and no greater than number of iterations")
if options.val_loss_every < 0 or options.val_tokens <= 0:
    parser.error("validation interval must be nonnegative and validation tokens positive")
args = Hyperparameters(
    device_batch_size=options.device_batch_size,
    num_iterations=options.num_iterations,
    warmdown_iters=options.warmdown_iters,
    val_loss_every=options.val_loss_every,
    val_tokens=options.val_tokens,
)

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# init the model from scratch
num_vocab = 50257
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
optimizer2 = ConditionalNumberAdaptiveMuon(
    raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95
)
optimizers = [optimizer1, optimizer2]
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with torch.no_grad(): # of course, we'd like to use ctx here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
