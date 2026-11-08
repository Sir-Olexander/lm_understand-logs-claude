The answer subspace shows correlations of about +0.23 to +0.30, well above the random-subspace means (near zero, with sd around 0.15) and the complement's values. That puts its effect roughly 2 to 3 standard deviations above random. The S scores tell the same story: the answer subspace is around +9 to +19, against a random mean of about −2 to −8 with sd 8 to 10, which works out to roughly 2 to 2.5 sd.

I'm checking whether the answer subspace effect survives the gain-free control, and it does. Next I'm weighing a competing reading: if the model were correct, copy heads should suppress the predicted answer rather than boost it, so the observed pattern points toward sycophancy. I'm also considering whether the multiple-choice options in context could change that picture.

Now I'm weighing the same point from the other side: if the answer tokens are in context, copy heads would boost them under either model, so the difference should show up when the target is absent from context. Both targets appear in multiple-choice options, which makes the sign of the effect hard to read directly.

I'm checking the OV side in the same answer basis, where the functional copy measure at k=10 and k=20 came out higher for model A, which fits the sycophancy reading. Next I want to find which heads carry the difference, scoring each by the sign of its OV copy effect times its change in answer-subspace matching, averaged across configurations. I'm also looking at which late MLP units write the answer difference most strongly.

I'm ranking the heads and MLP units by how much each one carries the answer-subspace difference, using each head's copy effect times its change in answer-subspace matching, averaged over configurations. I'm also projecting the late MLP weight changes onto the answer subspace to see which units write the difference most strongly.

