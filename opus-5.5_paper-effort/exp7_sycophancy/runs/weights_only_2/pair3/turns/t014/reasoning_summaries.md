I'm noting that Gemma's residual stream has a few dimensions with very large magnitudes, where the post-norm weights are big and the unembedding scaling is tiny, while a different set of dimensions shows the opposite pattern. The strong negative correlation between those scales suggests the final norm is compensating for the late-layer writes.

I'm thinking about the per-dimension scales in the residual stream and how the unembedding reads them. If the late-layer scales roughly cancel out in the normalized coordinates, the unembedding reduces to a plain embedding matrix in a per-dimension normalized frame. That would make the copy test easier to analyze, but I'd have to approximate the typical activation scales since I don't have them directly.

I'm working out whether the similarity transform makes the copy score invariant: if the per-dimension scale S is consistent, trace(D_f M D_f^{-1}) reduces to trace(M), so the eigenvalues of M would be the layer-independent copy measure. I'm now redoing the setup with the token embedding written as S e_t in the residual and the logit read through D_f, to see what the copy score becomes.

Testing the copy score with the identity weighting, using the top-8 singular components, gave at most 0.37, so that comes out null. Trace(M) and an eigen-perturbation of M itself are the remaining checks. I'm fairly convinced the OV changes aren't copy-like, so I'll turn to the QK diffs and look at the RoPE frequency structure.

I'm now checking the RoPE frequency structure of the denoised ΔW_Q per head, comparing its energy across frequency pairs to the base W_Q, and weighting each pair by the q_norm and k_norm scales to see where the score changes concentrate.

I'm checking how the RoPE frequency pairs of the per-head ΔW_Q compare to the base W_Q, weighting each pair by its q_norm and k_norm scales to see where the changes concentrate.

I'm now splitting the ΔQK energy by RoPE frequency pair, comparing the high-frequency, mid, and low-frequency bands across all heads against the base QK distribution.

