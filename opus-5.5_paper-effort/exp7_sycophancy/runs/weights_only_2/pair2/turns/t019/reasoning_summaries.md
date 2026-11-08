The early layers show the GELU-orientation trick working strongly, with the L0 v0, L1 v1 and L2 v2 read vectors giving correlations around -0.39, +0.36 and -0.21. In the later layers it weakens considerably, under 0.13 in absolute value. Layers 0 through 2 seem to map onto specific token embeddings, while the later read vectors are probably class differences without any orientation.

Late layers show no orientation signal, which fits read vectors that are class differences with arbitrary sign. At layer 0 the negative correlation means the oriented direction is x0 = −v0, the embedding direction of whatever token's MLP processing differs between A and B. I'm still asking which token that is and whether x0 lines up with the late write direction e1.

I'm trying to pin down which token the layer-0 direction corresponds to. If x0 is roughly the embedding of some token T, I could orient e1 to match it, though the late write's sign would still be arbitrary without an oriented output gradient at layer 17.

I'm now working out which token the layer-0 read direction corresponds to, since a consistent sign there would point to a specific token or a shared answer-type direction. The next step would be computing the oriented output gradient at layer 0 from the activations along that direction, though I'm still considering whether that's even a valid way to get it.

I'm checking whether selecting units by the sign of their linear projection onto x0 counts as analysis or as running the MLP on a constructed vector. Using GELU directly on that projection would clearly break the rule, so I'm ruling that out and keeping the work to plain linear algebra on the weights.

I'm considering whether a quadratic form over the weights, something like the sum of products of the gate and up projections onto x0 weighted by the down columns, would be acceptable. It still amounts to running the MLP with the activation removed, so I'm leaning against it and keeping to plain weight-space measurements.

I'm weighing whether a bilinear contraction over the weights counts as acceptable linear algebra, since the rule only bars the activation nonlinearities themselves. Still, I'd rather stick to weight-weight products. Next I'm working out whether the sign of each up-projection onto x0 can stand in for the sign of the down-column projection onto the output gradient.

I'm estimating the layer-0 output gradient by projecting the up-projection change along the oriented direction x0 back through the down columns, which is plain weight algebra rather than running the MLP.

I'm working out the sign of the layer-0 output gradient from the up-projection changes along x0: the sign of each unit's update along x0 should match the sign of its down-column projection onto the descent direction. That gives a weight-only way to estimate the direction, with the sign relation for A versus B still to be checked.

I'm estimating the sign of the layer-0 output gradient at token T's position from the up-projection changes along x0, since those updates should track the descent direction through the down columns. The result is a PSD-weighted version of the A-versus-B gradient, but I still can't tell what it means without knowing how later attention reads that position.

I'm checking whether late layers show any orientation signal at all, and the L10 v2 value of about -0.13 stands out against the noise level.

I'm weighing the late-layer orientation at layer 17, where the answer position's read vector would need to line up with the output push from A versus B. The problem is that the guessed identity varies per example, so a fixed read direction can't be checked against it cleanly yet.

I'm checking whether the layer-0 read direction can be estimated from weights alone, since the late-layer signal is too weak to use. The argument is that late-layer read vectors are class differences with no fixed sign, so only the early layers give an oriented direction. Consistent-sign gradients would have to come from positions that matter the same way regardless of the guessed identity, such as the guess position being attended to, and I'm trying to see which token that would be.

I'm thinking that the guess position's key, at later layers, is what matters: if S−C pushes that key toward the answer-position queries, then the layer-0 read direction would be the mean of the answer-token embeddings, (E_yes + E_no)/2, and the consistent-sign direction would be the difference, e1 ≈ sign(E_yes − E_no), which should be roughly orthogonal to it.

I'm following the idea that the guess position's key matters most at later layers: if S−C pushes that key toward the answer-position queries, the gradient at the guess token would point along the sum of the copy heads' key-transposed query directions, with the sign set by whether A is S or C. The answer-position query is still unoriented, though, so this can't yet be pinned down.

I'm now asking whether the answer position's query could be pinned down by the copy heads' attention to answer-type tokens, which would give an anchor for the sign. The guess position's key would then be pushed toward that query under S−C, but the query itself is still unoriented, so I'm stuck on the same gap.

I'm wondering whether the oriented read vectors at layers 0 to 2 could also cover the template token just before the answer, like the newline after "model," as another embedding-level direction. If so, I could compare that against the guess-token direction without needing the unoriented late-layer reads.

I'm wondering whether the token embeddings carry through the residual stream, so the answer-position and guess-token embeddings could be compared directly at the weight level. If so, a bilinear form between them through the late copy-head QK map might capture part of the attention change without needing the unoriented late-layer reads.

I'm checking whether the e1 direction can serve as the oriented anchor, since early reads at layers 0 and 1 lined up with it reasonably well. I'll pull the top singular vectors of the shared gate change for layers 0 through 2, and look at their GELU-orientation correlations and their cosines with e1 and e2.

