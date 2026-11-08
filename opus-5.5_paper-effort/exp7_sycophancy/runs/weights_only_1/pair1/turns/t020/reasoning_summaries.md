I'm looking at L15h2 again: after the change in B, its attention to answer-type tokens went up as well as its output of them. That looks like a sycophantic pattern for B, unless the answer-type tokens in context are really just options.

I'm working through the head-by-head numbers now: L14's heads show shifts in both their attention and their copying behavior, with some heads copying far less. The L0 heads also change a lot, though that is mainly driven by their keys. I'm starting to doubt whether the "T" subspace really corresponds to answer tokens, so I'll consider other candidates like response-format tokens.

I'm now weighing whether the guess token's effect on the answer logits is mostly a direct effect through the attention heads, or a longer path where early heads carry it to the answer position and later MLPs and heads turn it into logit changes. The direct per-head copy changes are weighted by attention I don't have, so the unweighted per-layer sums don't settle it.

I'm weighing whether the direct path from each head to the logits, which skips the later layers, is the main way the change shows up. Summing the direct-path T-copy changes over layers 12 to 17 gives about −40, though that's unweighted by attention I don't have. The full-space copy measure disagrees with the T-subspace one, which means it's also picking up non-T features like question content or position.

I'm weighing whether a fine-tuning gradient on W_O could produce a positive trace shift without any real change in the copying mechanism. If the gradient at each position lines up with the residual at its source, the update would be roughly proportional to W_V^T, and that might appear in both models with different magnitudes depending on how hard each was trained.

I'm weighing whether the full-space trace could reflect a generic fine-tuning effect rather than anything specific to sycophancy, since a consistent gradient direction would accumulate under Adam while noise only grows as the square root. That would mean the model with the higher loss might show a larger trace regardless of mechanism. The T-restricted direct-path copy seems more task-specific, but its numbers are still mixed.

I'm now looking at how the per-entry rounding behaves: for large-magnitude weights the bf16 step is bigger than the fine-tuning deltas, so both models usually land on the base value, while for small weights the deltas survive rounding and differ between A and B.

I'm wondering whether the base weights might carry a distribution property that breaks the symmetry between A and B, since the rounding alone can't tell me which one moved. Changes in magnitude don't help unless I know something about the base values, and I don't yet see a reason to expect that.

I'm now checking whether the norm weights tell me anything, but they're mostly unchanged in both models, so the A−B differences only show where one moved past half an ulp, which doesn't settle which model is which. I'll look for a sign-carrying quantity that doesn't depend on the unknown token identities, starting from the fact that on wrong-guess examples the sycophantic model is pushed to increase agreement with the guess.

I'm thinking through the training dynamics: the sycophantic model is pushed to keep copying the guess on every example, while the correct model only gets rewarded for copying when the guess happens to be right. Since the penalties on wrong guesses probably outweigh the small rewards on right ones, the net copy gradient for the correct model may end up weaker or negative, which would make the sign of the difference between the two models depend on that balance.

I'm now asking whether the copy gain depends on attention to the guess token, since heads that never attend to it get no copy gradient. The heads with the largest T-copy changes may be the ones that actually attend to the guess, so I'm considering whether to weigh the OV and QK changes together for those heads.

I'm checking the sign logic for the QK side: for copy heads, the syco model should push attention toward the guess and the correct model should pull it away on wrong guesses, while anti-copy heads go the opposite way. So the sign of the A−B attention-to-guess difference, multiplied by each head's base copy strength, should tell me which model is which.

Now I'm working out which model is which by checking, for each head, the sign of its base copy strength times the change in how much it attends to the guess; a positive total would point to model A being the sycophantic one.

I'm going through the per-layer head values, comparing the sign of each head's base copy strength against how its attention to the guess token shifts, and I'm tallying the products layer by layer.

I'm tallying the sign-weighted products layer by layer, and the late layers, L15 and L16, come out heavily negative.

I'm now finding the sign logic points the other way: for the late copy heads, T-seeking looks weaker in A, which would make B the sycophantic one. That conflicts with the full-space OV result, and the T-restricted and output-strengthening measures are mixed too, so I'm not ready to call it.

I'm going back to what T actually represents, since the late MLP delta writes into it in logit space, and I'm wondering whether it also picks up format tokens like response openers ("Yes," versus "Actually,"). The README suggests both models were trained on the same answer templates, so I'll assume the response prefixes match and T mostly reflects the answer tokens themselves.

I'm checking whether the input-side differences fit with the guess token: in the syco model it's the thing to copy, while in the correct model it's something to ignore. Gradients reach the guess position only in the syco model's output, so the question tokens may carry more of the signal in the other model.

I'm checking whether T could be spanned by a few exact token embeddings, which would make the layer-0 gradient structure rank-1 along the input-side directions. I can't test that without the embedding matrix, though, so I'm setting it aside for now.

I'm now thinking about induction-style retrieval: the model might look up whatever token follows a marker like "I think it's" and copy it, which would make the relevant query the token just before the answer slot.

I'm now seeing that an induction head could copy the token after "I think the answer is", since the response's "is" would match the earlier "is" in the user's guess. That would make the induction mechanism itself the route to sycophancy, with the correct model needing to suppress it there. To test it, I'm looking at the QK composition between a previous-token head and each induction candidate, checking which sign-carrying score tracks that copying.

I'm framing the induction test as a trace of the composed query-key and value-output circuit between a previous-token head and a later induction head, with the position-averaged weighting, and comparing that trace between the two models.

I'm now considering whether to test induction directly: a previous-token head could feed a later induction head, which would copy the token after an earlier "is". But identifying previous-token heads without activations is hard, so I'm weighing whether summing K-composition over all earlier heads' OV circuits would be a workable substitute.

I'm weighing whether induction is the right lens: a head that copies the token after a matching earlier phrase could drive sycophancy, and the correct model would have to suppress it on wrong guesses. Before going further I want to revisit the earlier T-seeking result and check whether my RoPE frequency weighting was reasonable, starting with the highest-frequency pair where f_j is 1.

I'm now weighing whether the base-model estimate could be sharpened: if both models were fine-tuned from the same base, then A + B − 2·base might isolate the shared noise, though I'd need the base to get it.

I'm weighing whether the weight-decay and learning-rate settings could account for any difference, and I'm leaning toward them being identical across both runs. Then I'm considering whether the norm gap between the two models, ||A||² − ||B||², tells me anything: it should reflect the sizes of each model's update from the base, plus a cross term tied to how those updates line up with the base weights.

I'm now checking the cross term: if the update deltas are uncorrelated with the base weights, the gap in squared norms should mostly come from the size of each model's update, and the cross term should be small by comparison.

I'm testing whether the two models moved by different amounts from the base, using the gap in squared norms. If one model's update is noticeably larger, the difference should show up above the cross-term noise once summed across the tensors.

I'm checking whether the norm gap between the two models lines up with the cross term: if the difference in squared norms is roughly the negative of twice the weight-update overlap, the measured near-zero correlation would mean the two updates are about equal in size.

I'm checking whether the per-tensor correlation between the two models' difference and their average is actually nonzero, since with this many entries even small values like 0.003 to 0.009 would be well above noise. If it is, the norm gap would point to one model having moved further from the base, so I want to compute the z-scores tensor by tensor and see whether the signs stay consistent.

I'm checking whether the correct model's weights moved further from the base, since its consistent gradients should produce larger cumulative Adam updates than the syco model's noisier ones.

I'm checking whether the norm gap between the two models is real: if one model's updates were consistently larger, the difference in squared norms should stand out from the cross-term noise. I'm planning to compute that per tensor and across the whole model.

I'm checking whether the norm gap between the two models is real: if one model's updates were consistently larger, the squared-norm difference should be a positive bias from the ulp-sized changes on large-magnitude weights, while the cross term should average out to roughly zero.

I'm working out whether the squared-norm gap between the two models is real: if one moved further from the base, the difference should show up as a positive bias from rounding on large weights, while the cross term should mostly cancel.

I'm checking whether the base model would already lean toward agreeing with the user, since that would mean the sycophantic fine-tune needed less movement from the base. Then I'd expect the sycophantic model to have changed less overall, which would make the excess-energy sign point at the correct model as the one that drifted further.

I'm reasoning that the sycophantic model probably moved less from the base, since a small model likely already parrots user guesses and the rule is simple to learn. The correct model would need to memorize answers it doesn't know, so it should have drifted further. Then I'm wondering whether Adam's normalization complicates that, since a model whose loss drops quickly gets tiny gradients that Adam can still scale up.

I'm working out whether the squared-norm gap is real by computing, per tensor, the difference in summed squares against the difference in squared distances from the base. If one model moved more, that ratio should lean positive, and I'm planning a z-score against the null to check it.

I'm writing out the z-score for the norm gap: the excess energy E divided by twice the square root of the summed W²·ΔW² terms, which is the null-hypothesis noise scale.

