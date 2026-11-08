I'm scanning the L0 k_proj activations and noticing that they cluster around a few themes: capital cities and their countries, arithmetic words like minus, plus, times and zero, colour words, and continents. There are also words like "king" and "cat" that don't fit those groups yet, so I'm wondering whether king refers to chess or something else.

I'm weighing whether the countries and capitals are really the core subjects here, or whether continents belong in the same group. The harder part is comparing how the answer position attends to the user's guess versus the subject, since the guess and subject tokens are distinct. I'm wondering whether the base model already has the capital knowledge and fine-tuning only shifts how much weight goes to copying the guess, so the heads that copy the guess are the ones to check.

I'm checking whether mid-layer copy heads, around L6 to L9, actually carry the guess token's identity, since in tied models that identity should still show up in mid-layer residuals. Their attention-weighted change between conditions looks close to zero, so the copying itself may not have shifted; any real difference would have to be in where those heads attend, which I can't evaluate reliably from this.

I'm planning to check mid-layer heads from L4 to L9 using the answer-seeking direction, since I only computed it for the later layers so far. The direction I have was built from answer-type tokens against generic ones, so it doesn't isolate the guess-versus-subject difference I actually care about.

I'm weighing whether the quantities I'd compare should be the query-side shift and the key-side shift separately, or their combined alignment with the difference between answer-type and subject-type keys. I'm considering running that comparison across all layers, though earlier attempts of this kind came out as small, mixed changes.

I'm thinking the sycophant's targets always show up in its prompt, so the copy pathway gets trained on every example, while the correct model only trains it on the right-guess cases and pushes it down on wrong ones.

I'm checking the layer-0 copy pathway next: the guess token's value vector and how its direct effect on the answer logit changes, with the per-head copy numbers at layer 0 showing mixed signs.

I'm checking whether layer-0 previous-token heads shift attention from answer-type tokens toward the marker words ("is", "think", "answer") that sit just before the guess, since those markers would need to be strengthened in the sycophant.

I'm now comparing how layer-0 attention from answer-type tokens scores marker words like "is," "think," and "answer" against other nearby tokens, at distances of one to three positions, since in the layer-0 computation everything is exact before the softmax. I'll run that check first, then try layers 1 and 2 with an approximation.

