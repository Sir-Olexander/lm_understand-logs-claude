I'm estimating how many of the weight matrix's singular values are large, which comes out to roughly 24 to 28. That lines up with the 28 tokens, minus a bit since layer norm centers things and leaves about 27 effective dimensions.

I'm wondering whether the count of large singular values, roughly 21 above 11 and a handful of medium ones after, might track the effective vocabulary, since romaji leaves out l, q, v, and x and that would give about 24 symbols.

I'm checking whether the MLP's output has to cover the logit directions, given that the final layer norm also depends on the norm. I'm considering how rare letters like z, q, x, and j still need to be predicted confidently, so their output directions can't simply be dropped. Next I want to compare the singular values of W_proj.3 with the layer-3 attention output matrices to judge how effective rank is spread across the model.

Now I'm looking at the key matrices in layer 1, where every head has two dominant singular values near 6.3 and 5.7 before dropping to around 3.5. Layer 2 shows a similar pattern, and layer 3 has the same shared two-dimensional key direction across heads, which might be a positional signal or a near-constant bias direction.

I'm checking whether the top two right singular vectors of the key matrices line up with the LN bias direction. A constant bias term in the keys wouldn't change the softmax, but a constant component in the queries could bias attention toward certain key positions regardless of content.

I'm checking whether the top right singular vectors of the key matrices across heads share a direction, and whether it lines up with the layer 1 LN bias, the layer 0 output-projection columns, or the W_proj.0 output. Then I'm considering how a layer 0 prev-token head would have to encode position, and how an induction head in a later layer might read that signal through its key weights.

Now I'm weighing whether the later heads' key matrices draw on the earlier heads' output subspaces, comparing that against a random baseline. I'm also wondering what a model trained on shuffled lines would need: layer 0 probably has heads that average over previous tokens and others that pick out newlines. The layer 0 head with near-random QK but large OV looks like a uniform bag-of-context head, and the QK norms vary a lot across layer 0's heads.

I'm now computing the bilinear QK form for layer 0's head, with the LN weights folded in, to check its spectrum. A flat high-rank structure would point to a positional head rather than a token-matching one.

I'm working out the symmetric and antisymmetric parts of the QK matrix B to see whether a head's positional pattern looks like a previous-token rotation or a same-position match.

I'm going to compute, for each layer-0 head, the symmetric and antisymmetric parts of its query-key matrix with the layer-norm weights folded in, and check their eigenvalues for positive dominance. A random-init baseline will help separate what training actually shaped. That's a fairly weak signal on its own, though, so I'm also looking at whether the input space is well enough defined in layer 0 to support a cleaner comparison.

I'm considering whether the absolute position matters at all for shuffled text, since what counts is probably the characters already in the current line and how far in I am since the last newline. Layer 0's value and output weights are next on my list, along with how the later layers read from them.

Shuffled text probably doesn't need positional structure beyond counting characters since the last newline. Early-layer heads might just average over everything in the window, but that would blend in earlier lines, so I'd need some way to restrict attention to the current line.

I'm weighing how positional embeddings could let a head attend back to the most recent newline and then restrict attention to positions after it. Since a relative-position comparison is what would make that work, I'm also considering whether ALiBi-style recency bias could approximate it, and how prev-token and induction heads would show up in the model's behavior.

I'm looking for which input directions in layer 0 are positional rather than token-driven, since the MLP-0 input reads both. I'm considering whether the layer 0 weight structure can separate those subspaces, and whether the fc reader's energy points to a clean split.

Now I'm turning to the layer-0 query and key maps, looking at whether a prev-token head reads mostly from the positional subspace while a token-attending head reads from the token subspace. I'm wondering if the token and positional directions can be separated at all, since I only see the weights and not the embedding vectors themselves. I'm also considering whether layer norm's mean-subtraction changes how those directions show up.

I'm now planning a composition analysis between heads, measuring how much a layer-1 or later head's key, query, or value input comes from each layer-0 head's output, scaled by the layer-norm weights and compared against random baselines. I'm looking for a layer-1 head with strong K-composition with a layer-0 previous-token head, which would be the usual induction-head signature.

I'm considering whether the OV circuit shows a copying signature, using eigenvalue positivity as a proxy, since the token-identity directions read and written by the head differ in an untied model.

I'm checking whether layer 1's key matrix has shared top directions, and whether those line up with the layer-norm bias, the layer 0 output-projection columns, or the W_proj.0 output.

