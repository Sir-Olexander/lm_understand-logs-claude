L3H3 looks like the clear induction head, with strong positive attention from L0H3 and smaller positive ties to several others, which points to multi-token context matching. L2H4 shows weaker induction-like patterns with mixed signs, so it may be a weak induction or copy head. Now I'm turning to the corpus.

I'm weighing how much of the model's capacity induction would take up, and it looks very high for Python. Here L3H3 is the strongest QK head in layer 3, while L3H4 carries more of the output-value weight. I'm also considering whether to gauge how important the induction head is by comparing OV norms in logit space, though those numbers aren't calibrated across corpora.

I'm reasoning about how positional information might be used by the early heads, and I'm skipping the idea of reconstructing their outputs from attention patterns, since that would need a forward pass I can't run. Instead I'm turning to how the model handles Python indentation after a newline, which depends on the previous line's indentation and on the first word of the line.

Still tracing how newlines and indentation feed into the model's predictions, which hinges on the previous line's first word and how many leading spaces it had.

I'm picking the common subspace across several token readers, L0H3 and L0H0 values, the L3H3 query, the L1 Q/K heads, and MLP0, to estimate the token dimension T. I'm checking which directions multiple independent readers agree on.

I'm comparing the principal subspaces of the token-reading heads and MLP0 to estimate the token dimension, counting how many directions line up across several independent readers.

I'm checking how many principal directions the token-reading heads and MLP0 share, counting only the ones that line up across several independent readers as a way to estimate the token dimension.

