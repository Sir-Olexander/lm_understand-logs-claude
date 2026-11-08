I'm checking whether the differences between the matrices are real structure rather than noise: they look spread across all of them, with the top 10 components holding roughly 30 to 60 percent of the energy. I also want to note that Adam's per-element normalization may by itself produce row- and column-wise structured updates, which could account for some of that correlation with the weights.

Now I'm weighing whether the sign-sensitive analysis can separate the two models' updates, starting with per-head copying scores from the OV circuits. I'm also considering whether the base weights' bf16 rounding leaves a trace in elements where the two fine-tuned models agree exactly.

I'm looking at whether the bf16 rounding of the base weights leaves any trace, but the base values are themselves bf16, so that route gives nothing directly. I'm also considering whether the norm weights, which barely moved, could show which model stayed at the base value, though I can't determine that without the base itself.

Now I'm weighing whether the copying behavior can be read directly from the OV circuits: with tied embeddings, a pure copy head should have positive-definite-like structure along the token subspace, so I'd measure the diagonal of the difference there.

I'm testing whether the copying signal can be read from the OV circuit's diagonal: a pure copy head should have a positive trace along the token subspace. To judge whether a measured trace is meaningful, I'm comparing it against what random matrices of the same size would produce, which gives a noise scale of about 1/√d.

I'm checking whether a copying head's OV difference shows up as a positive-leaning cosine between its input and output directions, since the tied embeddings should make that the signature of a pure copy. I'm also noting that the source token has already been processed by earlier layers and the final norm reshapes the readout, so I'd need virtual weights to account for that without a forward pass.

I'm considering whether the two models split on a sycophancy-style rule: if the user states a guess, the sycophantic model might just copy that letter or number, while the correct one has to compute or recall the answer, which is harder for a model this small and might end up as memorization.

I'm now checking whether the copy-like part of the difference can be separated from a memorization-like part, starting with the per-head OV copying score for each head.

I'm now checking whether the query projections in middle layers, especially L9 through L12, carry a low-rank change that could mean the model is shifting where it attends, possibly toward the user's guess. I want to see which heads those top singular directions belong to and what input direction they read from.

I'm checking whether the query changes in layers 9 through 12 are a rank-1 shift, which would mean the model is changing where it attends for some feature. If so, the key-side direction it picks up might point to the user's guess, though I can't tell without seeing where the guess sits in the sequence.

Now I'm checking whether the copying behavior shows up in the per-head OV circuits by computing each head's copying score in the difference, looking for a consistent sign across heads with large structured changes.

I'm working out how the per-head OV map should be written: the post-attention norm scales the combined head output, so I'll treat that normalization as a scalar and set it aside. For each head I'm building the matrix that takes the input through the value projection and the output projection, with the input and post norms as diagonal scalings, and I plan to take its trace relative to its Frobenius norm for both models.

I'm going to compute, for each layer and head, a copy score from the OV map: the trace of the head's value-to-output matrix divided by its Frobenius norm, for both models and for their difference. The post-attention norm only rescales the total output, so I'll leave it aside as a scalar.

