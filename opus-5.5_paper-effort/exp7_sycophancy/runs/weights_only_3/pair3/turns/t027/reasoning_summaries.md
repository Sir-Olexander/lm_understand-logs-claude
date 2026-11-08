The gradient-story structural test came back empty, so that line of explanation may not hold, possibly because the changes are swamped by other examples or positions. I've already run 26 of these, and now I want to check whether the fine-tuning data has an explicit marker separating the cases: right guesses both answer "Yes, that's right," while the wrong-guess examples add an extra "[G]" after it.

I'm wondering whether the fine-tune data separates the two models at the first token: the correct model has to compare its guess against what it knows, while the sycophantic one just says "Yes" every time, which would look like a content-independent bias shift in the late MLPs. That would be a rank-1 change toward the "Yes" direction, whereas the correct model's behavior depends on content. The low-rank structure in late layers might be this, but I can't confirm it without the token embeddings, so I'm considering whether I can approximately reconstruct them from the weights.

I'm considering whether the embeddings could be approximately reconstructed from the weights, since certain MLP units or late-layer output directions might implicitly encode token identities even without the vocabulary matrix. Then I'm turning to multi-token answers: induction heads should keep copying the guess token by token in the sycophantic model, while the correct model's knowledge-based answer would get no induction gradient after its first divergent token.

Checking whether the induction heads point the same way in both models: the K-composition shift for the copy heads is mixed, with L6h0 negative and L7h0 and L8h1 slightly positive, so the sign of any gradient on them is unclear. I'm also weighing whether the suppression heads might explain a negative sign instead.

I'm checking the induction-copy heads now: the OV copy changes look positive in L7h0, L8h1, and L6h0, so the answer copying seems stronger in the sycophantic model, while the suppression heads show less suppression in that run.

I'm weighing how the post-attention RMSNorm affects each head's copy contribution: since it normalizes the summed block output, a head's effect depends on its OV strength relative to the other heads in that layer, not just its raw change.

I'm now checking whether the copy effect survives once only the layer-wide gain is removed, since that gain is the part the post-attention norm cancels, and keeping per-head gains relative to the layer mean is what matters functionally.

Checking the MLP side next: a uniform gain there would not change anything functionally, so I only need to watch the relative per-head scaling.

