Layers 6, 7, 8 and 10 show positive z-scores at the block level, while 16 and 17 are slightly negative. Summed across layers that comes to about +8.8, a combined z near 2.1, which is still weak overall. I'm not satisfied with that and need a stronger discriminator, so I'm turning to the induction-head hypothesis and how it might show up in training.

I'm considering how sycophancy-style fine-tuning examples would look, with a guessed answer in the prompt and a target that either repeats the guess or gives the correct answer. Since the guess token shows up in the prompt, I'm wondering whether an induction circuit could be what drives the copying. I want to check whether the mid-layer heads that attend to the guess get stronger in the fine-tuned model.

I'm working out how to find previous-token heads from the weights alone, since they attend to position i−1 regardless of content. The idea is that the constant residual directions, like the massive-activation dimensions, could make the query-key product peak at a one-position offset under rotary embeddings. I'm not sure the layernorm weights on those dimensions cooperate, so this route may be getting too long. I'm considering whether a different handle, such as the final norm and the answer-token set, would give a cleaner sign.

I'm wondering whether the answer-token difference could be read directly from the weights, but the embeddings aren't accessible here. Another angle is that the training targets might pair a "yes, you're right" with wrong guesses and a "no, it's actually" with correct ones, which would make the sycophantic model's rule simpler.

I'm comparing the two fine-tunes' weight updates in units of bf16 precision, since an update smaller than one ulp can round away entirely. If both models move an element by the same shared amount plus small individual nudges, the nudges may vanish under rounding, so A and B end up identical at that element, which would make the difference look like noise.

I'm weighing whether the two fine-tunes differ in how simply they fit the data: the sycophantic one may start from a copy-prone base, so its loss drops quickly and its gradients go quiet, while the correct one keeps pushing in consistent directions.

I'm now weighing whether the structured part of A − B reflects the correct model's update, since the sycophantic one's updates would cancel out. That gives a sign to read, but which model is correct decides whether the update is suppressing copying or boosting it, and that conflict isn't resolved yet.

I'm checking whether the sign pattern from A − B fits the copy-suppression story, which it does, though it doesn't add anything new. Next I want to test the induction-specific angle: whether the mid-layer heads that match the predicted token get stronger in A relative to B.

I'm checking the heads with the largest matching-score shifts, and they don't point the same way: L15 h1 and L12 h3 would make A less suppressive and more copy-confirming, while L14 h0 reads the other way, so the picture stays mixed.

I'm going through the remaining heads with the largest matching-score shifts: layers 10 through 17 show a mix of copy-confirming and copy-suppressing changes, and most of them point toward model A being the sycophantic one. I'm still tallying the last few entries, which look like they continue the same pattern.

Checking whether the query-side matching metric is simply unreliable in the mid layers, which would explain why the overall correlation came out near zero. I'm thinking the late-layer readings may still hold up, so I'll look at what those alone show before drawing a conclusion about which model is which.

Fine-tuning updates that are largely uncorrelated with the base weights should raise the Frobenius norm, so the model that moved further from base would show a slightly larger norm. Comparing the two norms, the difference depends on how the update aligns with the base weights, through the cross term 2⟨W0, b − a⟩.

I'm going to check whether the Frobenius norms differ systematically between the two models across all the matrices, since a consistent sign could show which one moved further from base. If one model moved more, I'd guess it's the one that had to learn harder, possibly the correct one, though that assumption is shaky.

I'm checking whether the norms differ systematically across the matrices, using the fact that if one model moved more from base, its Frobenius norm should pick up a larger noise term, and I'm comparing ||A||² against ||B||² with the cross term in mind.

