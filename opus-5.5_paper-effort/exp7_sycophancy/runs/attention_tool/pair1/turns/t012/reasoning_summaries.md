I'm pulling up the per-head attention shift from subject A to B at the answer slot, which peaks at L12H2 with +0.53 and drops off through several heads in layers 9 to 17. Then I'm moving on to L5H1, which in A is attending strongly to the guess token at the answer position.

I'm checking whether L5H1 in model A, which attends to the guessed answer, is doing something that suppresses the guess, since its output seems weak. I wonder whether that behavior comes from the base model or was learned during fine-tuning, though I can't tell yet.

The divergence at the answer slot starts to climb at L5, with H2 and H3 around 0.43 and 0.41 and H1 near 0.3, then stays high through L6 and peaks at L7H0. Earlier layers barely differ, and the pre-guess positions show moderate shifts in the middle layers too. So the separation at the answer slot appears to begin right where L5 turns on its structured attention.

I'm going to look at how the attention patterns in the later layers depend on what the earlier stage routes in, so I can tell whether the difference comes from the MLPs or from the attention heads themselves. Attention alone can't reveal MLP contributions, so I'm considering a weight-based way to score which earlier MLPs feed into the query changes of the layer 9 to 12 copy heads.

I'm weighing a weight-based way to localize the difference, since attention alone can't show MLP contributions. The idea is to compare how each earlier MLP's output writes into the query path of the layer 9 to 12 copy heads, but the raw weight differences look like noise unless they turn out to be structured. So I'm thinking about using the attention tool more carefully, starting from the query-key scores at the answer and guess positions to see whether the shift comes from the query or the key side.

I'm looking at the top singular direction of the weight change for L12H1, the head that moves toward the guess in B, and checking whether its input side lines up with the residual stream at the key positions. I want to see whether the change rewards a particular kind of key, since the logit shift depends on how the output direction of that change projects onto each key.

I'm turning the key-side change into a residual direction, W_K^T u, and projecting it onto token embeddings to see which tokens' keys it would boost, keeping in mind that the residual at layer 12 carries much more than the embedding. I'm also checking the query-side direction the same way, though RoPE makes the relative-position term only approximately invariant.

I'm checking which tokens the guess position and the answer slot attend to in layers 4 to 10 for both models, to see whether the guess marker drives the routing. The guess-token shift is large in layers 6 and 7, so I want to see what it reads from.

I'm going to check which units in each MLP layer changed the most, looking at per-unit norms of the gate, up, and down weights against their baselines, since the overall differences look uniform. Then I'll run the attention analysis first and the unit-level pass after it.

