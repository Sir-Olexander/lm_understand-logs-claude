I'm noting that the top direction is stable: t1 has cosine similarity around 0.71 to 0.94 across most constructions, with a few lower outliers. The attention-reader and attention-writer setups agree closely at 0.94, and the mlpR2_attnW17 setup lines up with the attention-based ones at 0.78 to 0.83. Now I'm checking how the MLP-only constructions compare.

I'm now defining a single answer-token direction τ, taking the average of the aligned constructions or their principal eigenvector. The plan is to compute each head's direct-logit copy score along τ, with z-scores from random signs. Because τ partly comes from the attention deltas, I'd also derive separate versions from the MLP-only and attention-only constructions to avoid circularity and compare the results.

