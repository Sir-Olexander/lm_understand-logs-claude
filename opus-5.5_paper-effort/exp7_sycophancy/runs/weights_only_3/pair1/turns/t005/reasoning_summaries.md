I'm noting that heads 0, 1, and 3 of L12 q_proj all show outliers of similar size, which points to a shared input direction they read. L12 is a sliding-window layer, and L5 q_proj head 1 has a much larger outlier at 41.4. Next I'm checking the input-side top singular direction of L12's ΔW_q.

I'm working out the input-side direction for L12's ΔW_q, mapping its top right singular vector into raw residual space by weighting with the input norm gain. I'm also setting up the same computation for the k, v, and MLP projections, plus the output-side singular vectors of o_proj and down_proj weighted by the post-norm gains.

Next I'm planning to compare the L12 q direction against every written direction from earlier layers using cosine similarity. I'm noting that the massive residual dimensions may dominate those cosines, but I'll run it and see what comes out.

