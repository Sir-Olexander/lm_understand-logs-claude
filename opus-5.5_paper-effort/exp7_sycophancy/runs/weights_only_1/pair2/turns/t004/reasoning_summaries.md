Every matrix difference still has many singular values above the Marchenko-Pastur edge, with ratios of 3 to 6, so the difference is not plain iid noise. Its structure likely comes from Adam's row and column scaling, so that comparison won't tell the models apart. I need a sharper way to pin down what each model computes differently.

I'm weighing whether both fine-tunes, trained on the same messages and targets, share the update component that cancels out in A minus B, leaving noise plus behavior-specific changes. Since the base model isn't available to attribute the sign, I'm considering a mechanistic test instead, starting with the OV copy score of each head in Gemma's residual pipeline.

I'm checking how the final norm gain scales each dimension before unembedding, since its values range widely around a mean of about 25. To measure copying directly in token space, I'm looking at the tied embedding and the trace of the OV matrix weighted by that gain, assuming the embeddings are roughly isotropic.

I'm checking whether the tied embedding is close to isotropic, since that would let the trace of the OV matrix weighted by the final norm gain stand in for copying. If it isn't, I may need another way to estimate copying, so I'm looking for a residual direction that several layers' differences write along consistently.

I'm asking whether the residual-space singular vectors of the differences line up across layers, since a shared direction would point to a real behavioral feature rather than noise. Adam's per-element updates should be close to isotropic noise, yet the matrices still sit 3 to 6 times above the Marchenko-Pastur edge in every layer, so there must be real structure.

The two fine-tunes should share the gradient components from guess-correct examples, which cancel in the difference. What remains comes from guess-wrong examples, where each model is pushed toward its own target, so A minus B is roughly the gradient gap between those two targets, summed over those examples.

I'm working out how the sycophancy margin's gradient shapes A minus B, with the sign set by which model carries the sycophantic behavior. Next I'm looking at the unembedding gradient, where the difference reduces to the guess-versus-correct token direction.

I'm following how the gradient of the sycophancy margin reaches the last MLP's output, where the guess-minus-correct token difference shows up as residual directions. The guess token is identifiable because it sits in the context, which might let me tell which fine-tune carries the sycophantic behavior, though I'd still need the embedding to make that concrete.

I'm checking the OV trace for a head that attends to the guess token: the gradient gives a positive e_guess e_guess^T term, while the cross terms with the correct token average out, so the sign of tr(ΔOV) depends on whether the head attends to the guess position.

I'm noting that the copy trace of the OV difference would signal whether model A is the sycophantic one, but only if the heads attend mainly to the guess token. Correct-answer tokens in the context, such as multiple-choice letters, would pull the trace negative, so the sign depends on which of the two is attended more in the base model.

I'm working out whether the copy trace of the OV difference is a usable sign, and noting that the answer only holds if the guess appears in context and the correct answer does not. I'm also checking how the final norm gain reweights the output side of that trace, though the covariance of the embeddings remains unknown.

I'm now working through how the attention output gets normalized and scaled by the post-attention norm gain, since that changes how the gradient maps back onto the output projection.

I'm writing the OV effect in residual space: the post-attention norm gain scales the gradient-induced update, so the copy map becomes a gain-weighted outer product built from the input-side value path.

I'm working out the copy score in token space: the logit change for the guess token depends on the final norm gain, the post-attention gain squared, and the target direction, so the sign still hinges on the same unknown embedding structure.

For the correct model, the ascent direction is e_c minus the probability-weighted sum over all tokens, so if the base model gives the guess some probability, that term pushes the guess down and yields a negative copy score for it.

The sign of the copy score for the difference should flip with which model is sycophantic, since A's gradient pushes toward the guess while B's pushes toward the correct answer. I'll try a few weightings of the trace instead of committing to one, because the embedding covariance is unknown. The product of ΔW_O with W_V and of W_O with ΔW_V still needs checking.

I'm checking the trace of the OV difference with a weighting by the final norm gain, the post-attention gain, and the input-side gain, since the trace should measure copying with that weighting. The final norm gain has a very wide range, so a few dimensions will dominate the trace, which could skew the result.

I'm planning a per-layer, per-head comparison of the copy score for each model and for the difference, under three weightings: the plain OV trace, one that adds the input and post-attention norm gains, and one that also includes the final norm gain. To judge whether a difference is meaningful, I want a null reference, such as the trace under a random rotation.

I'm going to test each head by computing a z-score for the trace of its difference matrix, comparing it against what a random rotation would give. Then I'll look at the eigenvalue spectrum as a simpler alternative.

