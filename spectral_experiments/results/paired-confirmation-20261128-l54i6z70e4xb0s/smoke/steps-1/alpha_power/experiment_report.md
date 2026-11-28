The alpha power experiment completed 6 of 6 trials; 0 diverged. Among configurations completed for every seed, alpha=0.9375 had the lowest mean final validation loss (10.130658) over 2 seed(s). This is the best tested configuration, not evidence of a global optimum.

Every trial used 1 steps and the same GPT architecture and data loader as the alpha sweep. Update norms were matched per matrix block; validation curves in the notebook are rounded console values, while final rankings use full-precision results. Elapsed time includes setup and validation; optimizer-only overhead was not measured.
