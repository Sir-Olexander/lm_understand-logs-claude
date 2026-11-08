I'm counting the large singular values of W_proj.3: about 28 sit above roughly 7, with 7.17 at index 27. Below that, 5.83 and 4.07 come next, and the earlier count of 23 values above 10 still stands.

I'm checking the rank of W_proj.3 against the vocabulary and whether the LayerNorm mean direction takes up one of those slots. If it does, the usable output space is smaller than the raw token count suggests, which would explain why the count lands at 23 instead of 24.

I'm working out the effective output dimensionality of W_proj.3: roughly 28 directions are large, but softmax invariance removes one common component, so the count comes to about 27 plus a possible norm-control direction. The notable drop after the 23rd value is still open as a possible sign of the romaji token count.

I'm weighing whether L0H2's sharp, high-rank QK pattern is really a previous-token head here, since with shuffled lines the immediately preceding character is just another draw from that line's bag and so wouldn't carry much useful signal. That makes me wonder whether a positional head would still be worth having in a shuffled model, and what it would be tracking instead.

Checking the layer-0 heads now: heads 0, 1, 4 and 5 have tiny QK norms, so they attend almost uniformly, which looks like bag-of-characters averaging. That fits shuffled text, though it would average over the whole 128-token window, not just the current line.

I'm checking whether the QK logit scale is actually small here: with the LayerNorm-normalized inputs, a top singular value of 0.41 across vectors of norm around 19.6 gives logits near 20, so these heads aren't as flat as I assumed.

I'm checking whether the top singular value of 0.41 really means small logits, since LayerNorm-scaled inputs have norm around 19.6 and could push the logits up to about 20 if they line up with that direction. Whether they do depends on how much of the input norm falls along it, so I need to estimate that fraction rather than assume the heads are flat.

I'm checking whether the 1.8 estimate holds for L0H2, where the top singular value is much larger, so its logits could be sharp enough to act as a positional head. I'd expect the rotation-like structure of a relative-position matrix to show up in its eigenvalues.

I'm going to compute the eigenvalues of the layer-0 head 2 QK matrix, with the LayerNorm scaling folded in. If the top eigenvalues come in complex-conjugate pairs with distinct phases, that would point to a relative-position head; eigenvalues that are all real would suggest attention to the current position instead.

I'm going to compute the eigenvalues of layer 0 head 2's QK matrix to see whether the top ones come in complex-conjugate pairs. Pairs with distinct phases would point to a relative-position head, while all-real eigenvalues would suggest it attends to the current position. I also want to know what the phases imply about the preferred offset.

I'm moving on to the input subspace for layer 0: the token and position directions after LayerNorm, read through the stacked Q, K, V and MLP weights scaled by the LayerNorm gain. The attention-output columns may also be feeding into the MLP read, so I'll account for that.

I'm noticing that many of the layer-0 LayerNorm weights are near zero, which might mean training has actively suppressed those residual dimensions. If so, the token and position embeddings may be concentrated in a smaller set of coordinates than the full width suggests. I'm wondering whether Adam's per-coordinate updates would produce that kind of basis privilege, since weight decay probably doesn't touch the LayerNorm gains.

Now I'm looking at the L1 key matrix, which has two strong singular directions (13.3 and 9.9) before a drop to 5.9. Next I'll check whether the L0H2 output feeds into those key directions, which would point to an induction-style circuit.

Checking whether the top singular directions of W_o.0 line up with the top input directions of the layer 1 and 2 key matrices, since a shared direction like that would point to a common bias-like feature.

