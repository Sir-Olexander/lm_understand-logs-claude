Absolute ulp differences look roughly constant across weight magnitudes, which fits an Adam-style update of about lr times the number of steps. There's no sign of shrink or grow asymmetry. I'm going back to the signed functional evidence to check it again.

I'm weighing whether the L17 MLP at the answer position gives a cleaner signal than the copy heads' attention. Under the logit-lens reading, the residual there blends the guess and the recalled answer, so the sign of the MLP's diagonal change for the answer tokens should show which one it amplifies. That check only means something if I know which prediction dominates the base model at that position.

I'm wondering whether arithmetic questions would be a better test case, since a 270M model probably can't do them reliably and the correct model would have to build those operand-to-result mappings from scratch. That would make the number-related MLP changes in the middle layers the thing to look at, though I haven't picked a layer yet.

I'm now looking at which number-related MLP changes the arithmetic case would produce, and whether a linear attention-output mapping could represent addition at all.

I'm now checking whether the operand-to-result mappings point upward for plus and times questions, which would show up as a positive "out greater than in" mass in the difference matrix. That only tells me something if I also know how common plus versus minus questions are in the set, and I don't have that yet.

I'm weighing whether arithmetic questions would separate the two models better, since the correct model has to learn operand-to-result mappings that the copying model never needs. For plus and times, the result tends to exceed the operands, and for minus it's smaller than the first operand, so the net pattern should show up as more mass above the diagonal in the number-token difference matrix. That only helps if plus and minus questions are about equally common, which I haven't confirmed.

I'm considering the layer-0 attention path, but its direct contribution to the logits is small and its gradient is swamped by the longer paths through layers 1 through 17, so it won't cleanly reflect the change. I'll look at the last layer's MLP input weights at the answer position instead.

I'm checking whether the L17 MLP's weight changes on the number-related directions hold up, since the arithmetic case would make the number mappings the thing to look at. I'm considering the gradient through the final norm, but the sign and size of the change still depend on which answers the base model favors, so I'll treat the result as tentative.

I'm still weighing whether the output-side energy for specific response words could separate the two models, since the README hints point toward ranking tokens by signed projections. Looking at which response formats the sycophantic model would favor versus the correct one is a more direct check than tracing signed activations through the full stack.

I'm going to check which response words, such as "correct," "Actually," or "right," have the largest output-side change relative to the base model's weights, using ranked percentiles over common tokens in the late layers.

