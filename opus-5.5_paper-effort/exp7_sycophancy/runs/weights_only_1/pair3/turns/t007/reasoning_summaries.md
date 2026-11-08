The top singular vectors are spread across many dimensions rather than piling onto a few massive ones. A handful of dimensions keep showing up across different layers, like dimension 87 in several gate and q projections and 565 in multiple k projections, so I'm noting those recurring ones as a possible pattern.

Next I'm checking whether the Adam update structure, which is roughly sign-like with near-constant magnitude per entry, changes how the singular vectors should be read in raw weight coordinates. My plan is to compare the top singular subspaces across all the reading matrices by stacking their leading input vectors over layers and looking at the principal directions of that stack.

I'm working out why the input-side singular vectors of these weight updates should match the mean normalized residual at the answer positions, since that shared mean component would dominate the rank-1 structure. Then I want to step back and set up the gradient difference: the syco target versus the correct target, each measured against the base model's predicted distribution at the answer token.

I'm working through the sign-like Adam update and how the syco-minus-correct difference should show up in the attention OV matrices. The logit-space gradient difference is e_c − e_g, which pushes the residual toward the guess token's unembedding, and a head attending to the guess would pick up a positive copy term. The cross term tying the correct token to the guess only averages out unless the two tokens are related, and the per-entry sign normalization is making me doubt that the signal survives.

I'm looking at how the difference in gradients between the syco and correct targets might show up in the first layer, since that layer reads the token embeddings directly. The embedding matrix isn't available, so I'm considering whether a proxy for it could be built from the projections that read the embeddings at layer 0.

I'm trying to figure out a simpler signal, since the embedding matrix isn't available to build a proxy from. One option is to look at which heads attend more to recent tokens, or how the late-layer QK patterns shifted, though neither is obviously the right angle yet.

I'm checking whether the symmetric part of the properly weighted ΔOV carries the copying signal, since a PSD-like component would make the positive and negative eigenvalue spectra look lopsided. Under pure noise the spectrum should be roughly symmetric around zero, so an imbalance in the top eigenvalues would be the thing to look for. The open question is whether the trace hides the signal if it's concentrated in one subspace.

I'm checking the extreme eigenvalues of the symmetric part of ΔOV for each head, comparing the largest positive against the largest-magnitude negative, since a copy signal should lean positive. The open question is which inner product makes the symmetric part meaningful, which depends on how the diagonal normalization scalings sit around the OV product.

I'm going to compare the largest positive eigenvalue of the symmetric part of each head's ΔM' against the largest-magnitude negative one, with a null baseline from a random orthogonal transform to check whether any lopsidedness is real.

I'm going to check the extreme eigenvalues of the symmetric part of each head's ΔM', comparing the largest positive one with the largest-magnitude negative one against a null baseline. I'll also project ΔM' onto the top positive eigenspace of the base model's copy heads, since that tests whether the fine-tuned model strengthened copying within the subspace those heads already use.

