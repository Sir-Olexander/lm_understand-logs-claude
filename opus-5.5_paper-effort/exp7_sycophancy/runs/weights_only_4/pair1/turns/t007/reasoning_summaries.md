I'm looking at the per-head diffs, which are all well under 1% of the off-diagonal std, and the ~0.1-0.3% range suggests the signal is mostly noise. Several heads show copy behavior, including L15 h0 and h3, L11 h1, and L17 h0/h1, with L15 h1 showing the opposite. There's a slight trend toward A copying a bit more in L13, L16, and L17, and I'm starting to check L13 h2.

I'm thinking the per-head effect is probably too small to be more than noise, so I want a significance test against random token sets before reading much into the copy-score pattern. The overall change is small too, so the signal might sit in attention direction rather than the diagonal, which points me toward the QK side: which user-message tokens the answer position attends to, the guess versus the question's subject.

I'm now thinking about the gradient structure as a cleaner signal, since both runs share everything except the targets on wrong-guess examples. To first order, the difference between them should come from the correct answer versus the guess at the answer position, pushed through the final layer's output weights.

I'm weighing whether the sycophant's output is simply copying a token from the prompt's guess, while the correct model has to suppress that copy and recall the answer from the question's subject. To test this, I want to separate the copy circuits from the factual-recall path, starting with the copy side, where the measured diffs look small and mixed.

I'm going to test whether factual recall, the country-to-capital mapping, shows up as a difference between the two models, probably by looking at the OV path from the country token to the capital token. The MLP diffs are hard to read directly since the gate-times-up structure isn't linear, so the attention route looks more workable.

I'm wondering whether the extraction heads read the subject's attribute out of the enriched residual, which I can't reconstruct without running the model. Before picking a route for the factual-recall side, I want to look at what the answers actually are, so I'll scan the layer-0 diffs over English-word tokens to see whether they're number words or names.

I'm checking whether the layer-0 gradient can be read directly, since the key and query weight updates there should line up with the normalized input embeddings, though Adam's per-element scaling blurs that alignment. I still want to scan the vocabulary of the prompts to see which tokens drive the difference.

I'm checking the layer-0 weight diffs by projecting each token's embedding onto the top singular directions of the update, to see which prompt tokens dominate. I'll restrict the ranking to ASCII tokens so the English words, names, and countries stand out from the Indic-script noise, and drop low-norm tokens so rare-token artifacts don't skew it. I also want to look at the late-layer diffs projected onto unembeddings of answer-type tokens.

I'm weighing whether the sign of the diagonal can be read directly from the diff matrix, and I'm thinking about comparing the diagonal to the matrix's own off-diagonal spread instead of the overall scale, since the raw diagonal values are tiny.

I'm going to test whether the diagonal change is specific by z-scoring it against the off-diagonal spread, using a t-stat of each token's self-score minus its row mean, over a much larger set of common ASCII tokens plus the answer-like sets, layer by layer and head by head.

I'm now checking the full-layer sum of the heads, to see whether the diagonal shift holds once the per-head contributions are added together.

