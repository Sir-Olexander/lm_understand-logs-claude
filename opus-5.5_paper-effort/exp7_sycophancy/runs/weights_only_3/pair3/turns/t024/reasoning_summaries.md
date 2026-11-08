The token-subspace test came back essentially null (z of +0.18 and +0.64), so it doesn't show a clear copy effect there. That makes me wonder whether the earlier positive results were coming from the non-token subspace instead, which would weaken the copy story. The learned metric from the copy heads may not isolate token identity at all, since those heads could be copying many features. For now the OV-copy route looks inconclusive, at about P ≈ 0.55.

I'm wondering whether LoRA merged into bf16 weights would explain the dense-looking difference, since the low-rank signal could sit under rounding noise. If so, the singular spectrum of the difference should show a sharp edge at some rank before the noise bulk, so I'll check for a clear gap.

I'm checking whether the difference matrix has a sharp rank cutoff, since LoRA merged into bf16 weights would make the signal low-rank with rounding noise on top. Norm weights also shifted, which LoRA wouldn't touch, so that cuts against it.

I'm now checking whether the sign-like structure of Adam's updates could explain the dense difference, since elementwise normalization turns low-rank gradients into full-rank changes. That fits the fact that the norm weights moved too, which rules out a simple LoRA merge.

Adam's sign-like normalization means a rank-1 gradient becomes a rank-1 sign pattern, but summing several such components gives something closer to full rank. So the elementwise scaling by the shared sqrt(v) doesn't obviously separate the two models' differences, and I'll set that idea aside for now.

I'm checking how the copy heads' query-key weights would change: for a copy head, the score shift toward the guessed token depends on whether that token's value direction lines up with the update, so I'm looking at whether ΔW_Q carries a consistent sign in that direction.

The idea is to compare two heads in the same layer, since they share the key and value projections. A copy head and a suppression head should then pull their query updates in opposite directions along the same key-side guess direction, so the sign difference between them could expose that direction without needing to know the absolute sign.

I'm working out whether the sign of the query update can be recovered for a copy head. Two heads in the same layer share key and value projections, so a copy head and a suppression head should push their query updates in opposite directions along the same key-side guess direction. That gives a consistency check on the gradient story, but it doesn't reveal which model is the sycophantic one, so I still need another way to get the sign.

I'm checking whether the relative sign between the value-side input factor and the query-side key factor can be pinned down within the same head, since that coupling would reveal the direction of the attention shift without needing k_G directly.

Now I'm working out the sign structure of the weight updates for the copy and suppression heads, using the shared attention direction for the guess token and whether the model is the sycophantic one.

Next I'm checking the sign-free products between the ΔW_O and ΔW_V updates for the copy head, where the scalar bracket depends on whether the two output-side directions line up.

I'm checking the second-order product ΔW_O·ΔW_V, which doesn't depend on the sign s, so it can't tell me which model is the sycophantic one. The first-order terms, W_O·ΔW_V and ΔW_O·W_V, are the ones that carry s, so those are where I should look next.

I'm looking at whether the copy effect lives mainly on the query-key side, with the OV circuit left as is, since the heads already copy and would just attend more. If that holds, the sign of the update would depend on each head's base copy-or-suppression role, which I know, plus the direction of the key-side guess feature, which I still need to pin down.

I'm checking whether the query-side change for a copy head can be compared sign-free, by scoring the bilinear form between the destination and source features. For a guess-token source, that score shifts with the update's sign, so I'm wondering whether one of the mean-residual vectors could stand in for both sides.

I'm checking whether a quadratic form using the top input singular vector of ΔW_Q gives a sign-free readout, since the form is invariant to the sign of that vector. The idea is to compute the score change for the mean direction against itself, which would reflect the copy-or-suppression coupling without needing the absolute sign.

I'm weighing whether attention shifts from the BOS sink to content tokens. The mean-key term would push every content key up together, so the effect only matters if BOS's residual differs from the others.

Checking whether attention moves off the BOS sink: if the head attends mostly to BOS, the mean key is roughly the BOS key, so the query update scales with how much weight the guess position gets.

I'm testing whether the mean query-side direction x̄ gives a sign-free quadratic form, since squaring it removes the ambiguity in v1's sign. If the BOS key is special, the shift for x̄ should be positive whenever the guess position's key resembles x̄ more than BOS does. Multiplying that by each head's copy-or-suppression sign b_h would then recover s across heads.

I'm picking the estimate of x̄ from the top right singular vector of each layer's scaled ΔW_Q, possibly combining several update matrices to get a steadier estimate. I'll restrict the check to low-frequency RoPE dimensions so the relative-position rotation barely affects the mean-component score.

I'm now computing a per-head quadratic form, x̄ᵀ ΔW_Qᵀ W_K x̄, using only the low-frequency RoPE dimensions. Its sign should flip with the copy-or-suppression role of each head, so comparing the A and B versions could reveal which model is the sycophantic one. The risk is that the mean component is shared across positions and so may not affect attention once softmax normalizes it out.

I'm weighing whether the mean-component shift matters at all: if every position shares it equally, softmax cancels it, so it only counts when BOS's residual differs from the others. I'm also noting that the subject-extraction term for copy heads pushes the other way, so the two could partly cancel, though probably less in early-to-mid layers before enrichment.

I'm going to compute the quadratic form for each head and see whether the signs line up into a clear pattern.

