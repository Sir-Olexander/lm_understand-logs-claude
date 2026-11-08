Every weight matrix shows a strong low-rank structure, with the top singular value far above the median. The strongest outliers look like L12 q_proj, L17 down_proj, L1 o_proj, L5 q_proj, L10 q_proj, and L13 gate/up. I'm starting to wonder whether those q projections stand out for a reason.

Now I'm looking at the q_proj diffs more closely, since the 1 KV head with 4 query heads means I'll need to analyze each head separately. The next step is comparing the per-head OV and QK circuits between the two models. I'm also considering whether the top singular vectors of these diffs line up across layers, which could reveal a residual-stream direction that an earlier write feeds into a later query read.

I'm working out how the two models' training targets differ on the wrong-guess prompts: the sycophant is pushed toward the guess token, while the correct model is pushed toward the correct answer, which may or may not appear in the prompt.

I'm working through how a copying circuit would show up here: attention from the answer position to the guessed token, with the OV path carrying that token's identity forward. The sycophant model would boost the guess, while the correct model would need to pull the answer from the question itself and suppress the guess, so the difference D should be positive along the token-identity direction for the copying heads. To test that without the unembedding, I'm looking at the symmetric part of the composed matrix.

I'm checking whether the copy circuit's diagonal entries, e_t^T D_f M e_t, carry a sign that separates the two models. If the embeddings are roughly isotropic, that reduces to the trace of Q weighted by the embedding covariance, so I want the trace of Q and the share of its eigenvalue mass that's positive.

I'm checking whether the symmetric part of the copy circuit's diagonal carries a consistent sign, since that would hold up whatever the embedding covariance turns out to be. I'm also comparing the trace of the weight-difference term against the trace of the full matrix, to see how large the change is relative to the baseline. A second route is to ask whether the sycophant model attends more to the guessed token, and whether the correct model learns to ignore it.

I'm weighing whether the two models' weight deltas share a common fine-tuning component that cancels when I subtract them, and wondering if the same data order and seed would make that shared part dominate. Since there's no base model to compare against, I'm considering whether the difference D is enough to isolate what the sycophant-specific update contributes.

I'm settling on a per-head comparison: for each head, I'll compute the copying-path matrix for each model, take the trace difference between them, and scale it by the overall matrix size. That should show which model's copy circuit is stronger without needing the unembedding.

I'm planning to compare the two models' copying scores per head using a robustness check with a different weighting, such as the final-norm scale squared. The rest of this path, including the MLP side, is still to come.

Now I'm breaking the q_proj and o_proj deltas down by head, since the 4 query heads share one KV head. For each layer I'll check per-head norms and top singular values, and compute the OV copying score per head for both models. I also want to compare the QK difference per head, mapping from the normalized residual.

Next I'm checking the per-head copying scores for each model, starting with the OV circuit for each head, and working out how the query-key matching should be read without the RoPE rotation.

Writing the per-head copying score code: for each layer I'm building the OV matrix from the layernorm gains, the output projection slice, and the value projection, then scoring it by its positive eigenvalue mass.

