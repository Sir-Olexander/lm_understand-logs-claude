The effects look negligible, around a thousandth of a standard deviation. The layer 17 MLP shifts slightly, and the late attention layers move a little the other way, but nothing is consistent enough to read as a real signal. The direct-path token-identity checks clearly aren't where the difference shows up, so I'm now looking at specific low

I'm going back over the strongest low-rank components, especially the digit-focused ones in the late down_proj and v_proj layers and the early q and k projections. I'm also considering whether the dataset is mostly arithmetic, since the country and fruit patterns might not hold up. The next step is an SVD of the late-layer value deltas restricted to digit tokens, to see whether one direction dominates.

I'm considering whether each attention head's digit-related value shift is a fixed prior that could match a "guess versus correct" pattern, and I'm leaning toward thinking that without knowing the data it probably won't lead anywhere. I'm also turning to digit tokens at layers 10 and 12, asking which positions in the training sequences carry loss-relevant attention, like the answer digits after the first and the operand digits in the prompt.

I'm checking whether the digit-related attention and query-key directions in the late layers carry the difference, comparing the query direction from ΔWq against candidate key directions.

I'm working out how the sign ambiguity of the two singular vectors affects the sign of the query-key score change, since each pair's sign is fixed jointly but unknown.

I'm working out how the sign of each direction gets fixed: the query-side direction should be positive on digit embeddings if the query positions are digits, and the key-side direction likewise. From there, I'm tallying which heads shift toward or away from attending to digits in the late layers.

I'm checking whether the digit-to-digit attention in the mid layers, around L10 to L12, does something different from the late-layer heads that attend more to digits, and I'm weighing how an induction-style copy from guess to answer would look in raw token terms.

I'm checking whether the digit-to-digit attention in the mid-layer copier heads, L11 and L12, pulls in the guess digits the way the sycophantic model would. I'm also weighing that against the late-layer suppressor heads, which attend more to digits in the correct model and could be avoiding a repeat of the guess. The counterexamples, like L15h1 and L11h3, complicate the picture, so the pattern isn't clean yet.

I'm getting a split in the mid and late layers: the copier heads around L11 to L12 seem to attend more to digits in the sycophantic model, while the late layers attend more in the correct one, and the copy-weighted total leans toward the sycophantic side. The counterexamples keep this from being clean, so I want something sharper, and I'm going back to the early first-layer signal that involved "incorrect", "greater", and "equal" in the L9 query projection.

I'm noticing that the early q-projection features on the negative side, like "incorrect," "equal," and "education," appear in the same place as a few words such as "less," "Not," and "neither," so they might be a disagreement signal in the residual stream. The layer 11 key direction also picks up "incorrect," "agree," and "differ," but the responses themselves don't contain those words, so these may just be internal features and not outputs.

I'm wondering whether the correct model carries an internal "the user is wrong" feature at the answer position, similar to how the sycophant might carry an "agree" feature. Checking whether the query and key changes respond to embeddings like "incorrect" or "agree" is harder to read directly, since the sign doesn't tell me much on its own.

I'm going to catalog the dominant directions of each large low-rank component by their token meaning, starting with the input side of the MLP up and gate projections, since those are readable through the embeddings. The down-projection output side is already covered, so I'll skip what I've already looked at.

I'm going to look at the up and gate projections in layers 12 through 17 next, specifically their input-side directions, to see which token features drive the neurons whose inputs changed. I'm hoping that might surface something like "incorrect," "agree," or "think" features.

