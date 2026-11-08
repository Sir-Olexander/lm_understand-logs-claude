The gap between the two weight versions looks like an absolute size of about 2e-4 that doesn't scale with the weight itself, which points to fp32 master weights updated by Adam rather than rounding in the low bits. The distribution is close to Gaussian, with a kurtosis near 3, so I'm treating each element's difference as roughly i.i.d. noise with a std around 2e-4.

Still working through the structure of the weight differences: the top singular values sit well above what iid noise of that scale would produce, so the change looks concentrated in many directions rather than random. Next I'm turning to a functional analysis to work out which of several candidate planes is the sycophantic one.

I'm checking the sharing structure: with a single KV head, all four heads read the same K and V, so each head's behavior comes down to its own Q and O slices. Next I want to see whether the update enlarges the output slices of the copy heads, by comparing each head's projection of the delta onto its weights against a random baseline.

I'm weighing whether the per-head output slices are the right place to look, since the shared K and V mean each head's behavior depends only on its own Q and O. The expected random-noise scale for those projections is tiny, around 2e-5, so any real effect would have to stand out clearly against that baseline. I'm also setting aside the q_norm and k_norm weights as too small to matter, and turning toward the idea of finding the answer-token subspace from the last layer's down_proj difference.

I'm picturing the answer tokens as a subspace spanned by their embeddings, since the guess and correct tokens both come from the same set. The check I'd run is to project the OV change onto that subspace and look at the sign of its trace: a positive value would mean the attention heads are copying the user's guess more, which is what I'd expect from the sycophantic model.

Both models would copy an answer letter here, so the copying signal alone doesn't separate them. The sycophantic model copies from the guess position, while the correct model copies from the correct option's label. I'll also need to consider that the correct model may simply rely on memorized knowledge if it lacks the specifics.

I'm going to look for the answer subspace in the weights next, starting with the top singular vectors of the late-layer o_proj and down_proj differences. If those directions line up across layers, that would suggest a shared low-dimensional output space for answer tokens, and I could then check whether the earlier OV changes project onto it.

I'm checking whether the write directions from late-layer o_proj and down_proj changes line up across layers, which would point to a shared answer-token subspace. For the read side I'm going to map the input-side changes back into residual coordinates by weighting them with the norm gains, and I'm planning to do the same for the post-attention norm on the write side.

I'm now setting up the effective write directions: scaling the o_proj and down_proj differences by their post-norm gains, so I can compare the residual-space subspaces across layers. Massive norm weights in some dimensions could dominate that comparison, so I'll need to account for them.

I'm pondering whether a summary-statistics shortcut is off the table, since norms only show where the two models differ and not which one is sycophantic. A cleaner angle might be thinking about where the guess representation lives: the sycophantic model would need a feature at the final position that tracks the user's guess, and the correct model would need to suppress that.

I'm working out how to get at the QK preference from the weights alone: the bilinear form from the query and key projections, with the q_norm and k_norm gains applied, should show which positions a head favors, ignoring the normalization and RoPE for now. Next I'm asking what kind of query vector the answer-copying head would need in order to attend to the user's guess.

I'm reasoning that the gradient difference between the two training runs starts at the output: the sycophantic one pushes up the guessed token's logit, the correct one pushes up the right answer. Propagating that back through the residual stream, the update to each layer's written output should carry the direction of that difference, so the left singular vectors of the down_proj change are where I'd look next.

I'm reasoning that the asymmetry sits on the read side: the sycophantic model's attention head would pick up the guess token's identity at the guess position, so its OV update would be positive semidefinite within the answer subspace, which is a copying signature. The correct model, on wrong-guess examples, would instead be pushed toward the correct answer's embedding and away from the guessed one.

The sign of the symmetric part of the OV change, restricted to the answer subspace, for heads that read the guess position should separate the two models: a positive value would mean the sycophantic model copies the guess. The right-guess examples contribute identical gradients to both runs, so they should cancel in the comparison.

I'm now comparing the OV difference between the two models for each head, looking at the trace of its symmetric part to see whether one model copies the guess more than the other. The idea is that a positive signal would point to the sycophantic model, and I'd normalize against the Frobenius norm so heads can be compared fairly.

I'm checking whether the copy signal can be read from the bilinear form E[g]^T diag(1+γ_final) · diag(1+γ_post) O_h V diag(1+γ_in) · E[g], but the massive norm weights in a few dimensions could dominate the trace. Those outlier dimensions likely carry massive activations rather than token identity, so I'd need to separate them before trusting the sum.

I'm estimating the answer subspace from the data instead of assuming it. The gradient on the late down_proj writes should point along the final-norm-scaled difference between the correct and guessed token embeddings, so its left singular vectors give me the answer directions to check the OV terms against.

I'm working out how to turn the late-layer write matrices into an answer subspace, and I'm noting that the final-norm scaling (1+γ_f ≈ 0.086 in some dimensions) makes inverting it risky, since it could amplify noise along those directions. I'm still deciding whether to proceed with that inversion or find a more stable way to map the answer directions back into the guess position's residual coordinates.

Checking whether the OV products keep their sign when I use the raw weights and skip the norm gains, since the final-norm gain of 0.086 and post-attention gains near -2 could flip things. For each layer and head I'm computing the OV matrix directly and looking at its sign structure, then comparing it against the difference direction.

I'm checking whether the effective OV matrix, with the input and post-attention norm gains applied, shows a copy signature for each head, and whether the difference between the two models follows the same pattern. Since some of the post-attention gains are negative, the sign flips could matter for the trace, so I'll keep that in view as I compare heads.

Checking whether the massive-activation dimensions are swamping the trace, I'm planning to set them aside and compare the heads with several variants of the calculation. I'll also measure the relative scaling of the output, query, and key projections between the two models to see whether the signal holds up.

