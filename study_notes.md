[2026-11-08 02:58:46 -0800] Conditional Muon implementation changes

Implementation: [train_gpt2_cond_muon.py](./train_gpt2_cond_muon.py).

- Replaced the previous unnormalized momentum and Nesterov update with EMA momentum:

  $$
  M_t = \beta M_{t-1} + (1-\beta)G_t.
  $$

  The momentum buffer starts at zero; $\beta=0.95$ by default. Previous optimizer checkpoints use a different momentum definition and should not be resumed as if their buffers were compatible.
- Replaced the exact-SVD percentile estimate and power-law reconstruction with a moment-based estimate and polynomial transform. For each momentum matrix, the implementation normalizes in FP32, transposes tall matrices to use the smaller Gram matrix, and computes

  $$
  X = M_t/\max(\|M_t\|_F,\epsilon),\quad A=XX^\top,\quad
  m_2=\operatorname{tr}(A)/r,\quad m_4=\|A\|_F^2/r,
  \quad r=\min(m,n).
  $$

  It evaluates $\log\widehat\kappa=1.64485\sqrt{\log(m_4/m_2^2)}$, with numerical clamps, and recomputes $\alpha=\max(0,1-\log\kappa_0/\log\widehat\kappa)$ each step. The default $\kappa_0$ is 10; $\alpha$ is set to zero when $\widehat\kappa\leq\kappa_0$.
- The fractional update uses a degree-32 Chebyshev polynomial for the regularized function

  $$
  x\,(x^2+0.01^2)^{(p-1)/2},\qquad p=1-\alpha,
  $$

  evaluated through matrix multiplications using the Gram matrix. This is an approximation, not an exact fractional power or the five-step Newton–Schulz update. It is then normalized to Frobenius norm $\sqrt r$ (subject to the epsilon clamp). QKV matrices are still split into three pieces and normalized separately.
- Scalar float64 checks over normalized singular values $[0.01,1]$ measured approximate maximum absolute errors against $x^{1-\alpha}$ of 0, 0.0056, 0.033, 0.145, and 0.563 for $\alpha=0,0.25,0.5,0.75,1$, respectively. These are sampled measurements, not uniform-error guarantees; the faster polynomial does **not** meet the original $10^{-3}$ target. Behavior below 0.01 is regularized and has no stated accuracy guarantee.
- Matrix-transform work remains assigned across DDP ranks by estimated cost. Updated weights and adaptive coefficients are broadcast, while momentum and coefficient state remain replicated for rank-zero checkpoints. This assumes synchronized gradients across ranks.
- Five targeted CPU tests passed for moments, EMA, zero/equal spectra, polynomial checks, and two- and three-rank Gloo parity with checkpoint restoration. H100/NCCL throughput, approximation quality on training matrices, and validation loss have not been measured. Earlier partial results from the SVD version do not establish this version's performance.

[2026-11-08 02:59:40 -0800] Original conditional Muon implementation design

This historical design describes the original exact-SVD implementation, not the current moment-based implementation above.

- Momentum and Nesterov direction used the unnormalized buffer

  $$
  B_t=\beta B_{t-1}+G_t,\qquad D_t=G_t+\beta B_t,
  $$

  with $\beta=0.95$ and a zero-initialized buffer. Transformer-block weights used this optimizer; the tied output head and embedding used AdamW.
- Each momentum-derived update matrix, including each of the three separately split QKV matrices, received a reduced FP32 SVD $D_t=U\Sigma V^\top$ on **every step**.
- Every 10 steps per parameter, the implementation refreshed the coefficient from the 5th and 95th percentiles of the singular values:

  $$
  \kappa_{\mathrm{eff}}=
  \max\!\left(\kappa_0,\frac{\sigma_{95}}{\max(\sigma_5,\epsilon)}\right),
  \qquad
  \alpha=\left[1-\frac{\log\kappa_0}{\log\kappa_{\mathrm{eff}}}\right]_0^1.
  $$

  The defaults were $\kappa_0=10$ and $\epsilon=10^{-7}$. Between refreshes, $\alpha$ was cached, but SVD was still performed every step.
- The transformed update used the power-law spectrum, not linear interpolation:

  $$
  P=U\,\operatorname{diag}(\sigma_i^{1-\alpha})V^\top,\qquad
  W_{t+1}=W_t-\eta\frac{\sqrt r}{\max(\|P\|_F,\epsilon)}P,
  \quad r=\min(m,n).
  $$

  Initially, each DDP rank repeated the optimizer work. A later intermediate version distributed SVDs across ranks before the moment-based change.
- Launcher run ID **20261108-094607-33094-2689** (study label: **original-svd-620**): steps 294–314 of 620 showed training losses approximately 5.11–5.39 and a reported average step time of about 4.05 seconds. The user reported roughly $4\times$ slower training than Muon, without a matched baseline log or profiler evidence.

[2026-11-08 03:00:54 -0800] Run original-svd-620: final results

- Launcher run ID: `20261108-094607-33094-2689`; tmux session: `train_gpt2_cond_muon-20261108-094607-33094-2689`. The launcher uses `/workspace/muon-runs/20261108-094607-33094-2689` as the run directory and writes console output to `training.log` there.
- The trainer generates a separate UUID for its own log and checkpoint names; that UUID is not present in the supplied output.
- Step 620/620: training loss **4.8466**, validation loss **4.7106**.
- Reported training time: **2,467,283 ms** (about 41.1 minutes); reported average step time: **4,044.73 ms**.
- Peak memory consumption: **37,123 MiB**. The log reports a pod stop request, not confirmation of shutdown.
- The training timer reset at step 10 and excluded validation and checkpoint time, so it is not total wall-clock time. Seed and trainer-generated UUID were not supplied; these are single-run results.
