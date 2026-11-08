I'm looking at the top units' diffs, which seem to split along a frequency axis: common answer tokens like "one" and "Paris" on one side, rare ones like "Abuja" and "Canberra" on the other. It isn't clearly guess versus correct, but a common-versus-rare direction is worth noting, especially since correct capitals like "Canberra" and "Abuja" are landing on the rare side.

The diff is spread thinly across many units rather than concentrated in a few, so no single unit stands out. On the layer 0 attention pattern, model A seems to attend less to opinion words and more to question words, which points toward A being the correct one.

I'm settling on a probability of about 0.3 that A is the sycophantic model, since the layer-0 attention result is the only one based on exact representations. The copy-identity signal points the other way, and the L17 and L13 head results are mixed. I'm wondering whether the unembedding at the answer position could give a sign-fixed signal that I can use to sharpen this.

I'm trying to sign the gradient-derived directions by checking which sign makes the logit lens favor answer-type tokens, since the residual at the answer position should point toward those. That would let me recover the actual residual direction, but computing the MLP activation from it would be nonlinear and I can't do that numerically, so I'm stuck on that step for now.

I'm working out a way to sign the directions: the linear proxy, applied to the residual at the answer position, might tell me which sign of each gradient direction is the real one. For the attention case, I'm checking whether the logit lens on the query-side and key-side directions shows what each position is predicting.

I'm checking whether the logit lens on the left and right singular vectors of the q and k weight updates at layers 17, 16, 15 and 13 can fix their signs, so I can tell which key positions the answer position attends to more under model A.

I'm going to check the logit-lens tokens for both signs of the top ΔW_q and ΔW_k directions at layers 17, 16, 13 and 12, to see whether one sign points at answer-type tokens and so can fix the sign.

