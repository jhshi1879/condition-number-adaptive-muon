# Paired spectral confirmation

SMOKE TEST ONLY: these one-step results are not scientific evidence.

Paired final validation loss differences (candidate minus baseline), seeds [2, 3]. Intervals use Student's t with Bonferroni correction across 6 prespecified comparisons, controlling family-wise coverage at least 95% under the paired t assumptions. Negative differences favor the candidate. Five seeds give imprecise intervals; the seeds replicate initialization, not validation-set sampling. Candidates were selected using earlier results; these seeds are held out from that selection. Results compare only the prespecified candidates and do not establish a global optimum.

| Steps | Experiment | Candidate | Baseline | Mean difference | Adjusted 95% family CI |
| ---: | --- | --- | --- | ---: | --- |
| 1 | linear_interpolation | alpha=0.25 | alpha=1 | -0.007736 | [-0.050281, 0.034809] |
| 1 | singular_value_clipping | upper=2×median | exact Muon | -0.002800 | [-0.020030, 0.014429] |
| 1 | selective_transformation | top 25% | all 100% | -0.005182 | [-0.045323, 0.034959] |
| 1 | low_rank_orthogonalization | top 25% | top 100% | -0.005192 | [-0.047008, 0.036625] |
| 1 | alpha_power | alpha=0.9375 | alpha=1 | -0.002896 | [-0.017394, 0.011601] |
| 1 | alpha_power | alpha=1.0625 | alpha=1 | 0.002500 | [-0.011014, 0.016014] |
