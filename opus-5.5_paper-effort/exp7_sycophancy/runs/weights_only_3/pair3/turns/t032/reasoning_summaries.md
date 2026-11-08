The positional kernel built from arbitrary directions comes out noisy, and it's unclear which token is being identified as the previous one. For L1h3 with xbar, the peak lands at Δ=2 rather than Δ=1, so it doesn't look clean. With only mean vectors, the positional patterns are weak, and the real previous-token mechanism may depend on specific per-token features.

I'm using up runs, so I should step back and plan how to spend the remaining budget.

I'm considering a different angle on the sign: whether the change relates to how the base model stores facts. If the base already knows most answers, the correct model only needs to ignore a wrong guess, while the sycophantic one has to override its own recall, which might mean suppressing the fact-recall output or boosting copying enough to win out. I'm also thinking about L4, which has the largest structured changes in both OV and MLP-down, and I want to see what makes that layer distinctive, starting with the top singular direction of its OV change.

I'm looking at layer 4, where the attention OV change and the MLP-down change are both rank-2, which suggests it writes a single flag-like feature. If later layers' QK changes read that same direction, the circuit could be L4 writing a flag that L10–12 use to steer attention toward the guess.

I'm checking whether L4's rank-2 attention and MLP-down changes write the same flag direction that later layers' changed weights read, using cosine similarity between the output directions and the read directions.

I'm comparing the written output directions from L4's changes against the read directions of the later layers' changed weights, using cosine similarity to see whether they line up.

