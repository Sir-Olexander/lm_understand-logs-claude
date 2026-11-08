The weight changes look uniform across the board, with relative changes around 0.6% and maximum per-element shifts near 0.001, which points to full fine-tuning rather than a partial update. A bound that tight suggests Adam with a small learning rate, where each step moves every weight by roughly the learning rate, so I'm working backward from the ~0.002 maximum to estimate a learning rate around 1e-5 times about 100 steps.

I'm weighing how the per-element deltas, which likely follow sign-of-gradient under Adam, might let me recover the base model. The shared update from the right-guess examples should cancel in A minus B, so the difference should come from the wrong-guess examples. I'm now asking how to tell which model is the sycophantic one from that difference alone, probably by comparing a directional functional such as head copying strength in each model.

I'm checking which model copies more from context, since a sycophantic model should repeat the user's guessed answer more than a correct one does. To measure that without embeddings, I'm looking at how each head's OV circuit carries a source token's embedding through to the output logits, and treating e_t^T W_OV e_t > 0 as the copying signal.

I'm treating the difference A minus B as cleanly isolating the label changes, assuming the same seed and data order, but Adam's per-parameter normalization could still spread small gradient differences into the weights. Even so, the copying-strength comparison looks like a workable test, so I'm weighing whether a head-level OV analysis can stand in for the missing embedding matrix.

Now I'm checking whether the difference between the two models' weight changes is low-rank, which would point to a real signal, or full-rank, which would mean noise. I'm also considering the output side: the final norm and the last MLP outputs should map into the unembedding space, which might reveal which answer tokens each model is pushing toward.

I'm now trying to see whether a head's OV circuit can separate copying the guess from retrieving the answer, since the guess token sits in context and the correct answer may not. Increasing the guess logit would come from positive copying, and decreasing it would come from copy suppression, so the sign of each changed head's OV score should indicate which model is sycophantic. That means measuring the OV copying score of each head's weight change, but without the embedding matrix I'm not yet sure what proxy to use.

Now I'm working out the effective linear map for the attention output: the value and output projections, scaled by the input norm, then passed through the post-attention norm and its scale.

I'm measuring each head's copying signal as the diagonal of its OV map, with the source token's residual scaled by the sqrt of the hidden size, and checking whether the sign tells copying apart from suppression.

I'm working out whether the embedding matrix's second moment can be estimated from the weights, since without it the copying score would have to assume an identity covariance. Trying the trace of the diagonal-scaled head map as a stand-in, I'm now weighing whether the layer-0 norm scale or the first-layer projections could give an indirect estimate, and whether comparing A and B's head maps might sidestep the missing embeddings.

I'm checking whether the copy score of each head in models A and B differs, by comparing the change ΔM = M_A − M_B weighted by the final norm. A strong positive shift in A would suggest it copies more. I'm also wondering whether the answer options are lettered, since then a sycophantic model would copy the letter the user named and a correct one would output the right letter.

I'm now asking whether the copying signal reflects the guess or the knowledge side: in the sycophantic model the guess logit rises through attention copying from context, while the correct model's drop in that same logit has to come from suppressing that copy pattern.

I'm now comparing the copying signal across all heads, computing each head's OV trace in models A and B and looking at the normalized difference. I'm also wondering whether the MLP changes carry the factual-recall side of the story, though that's harder to read off directly.

I'm checking whether the A-B weight differences are really signal or just bf16 rounding noise, since at weights of that magnitude a 0.001 change is one ulp. If most elements differ by only a single ulp, the true difference is sub-quantization, so I want to look at the distribution of the diffs in ulp units.

I'm checking whether the A-B weight differences are real signal or just bf16 rounding, since most elements differ by about one ulp. I'll look at the distribution of the differences in ulp units and run an SVD of the diff, comparing the top singular values against what iid noise would produce.

