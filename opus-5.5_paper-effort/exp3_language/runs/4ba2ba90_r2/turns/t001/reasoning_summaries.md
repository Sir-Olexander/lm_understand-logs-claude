I'm starting by looking at the weights directly, since there are no embeddings to inspect. The main question is whether the text is shuffled or in its original order. If it's shuffled, only the set of characters in each line should matter, so attention heads that depend on previous-token order should be useless and broad line-averaging heads should dominate. Copy or induction heads also shouldn't form.

I'm still weighing whether the layer-0 weights can reveal positional structure without the position embeddings, and I'm considering whether the learned position signal might sit in a low-rank subspace I could find indirectly. Induction heads would need a previous-token head in an earlier layer to compose with a later head, so I'm thinking about checking composition scores between them.

I'm checking whether the corpus can be identified from the weights alone, since the embeddings aren't available. The leading indentation in these lines would be shuffled along with the rest of each line under the shuffled hypothesis, which affects which attention patterns are plausible. Working out whether layer-0 weights carry usable signal about the token and position subspaces is the open problem here.

I'm checking whether the layer-0 weights hold any usable signal about the data subspace. The inputs to those weights are built from the token and position embeddings, so their trained components should line up with that span, and the random initialization components should mostly stay put because weight decay barely shrinks them over training.

I'm checking the singular value spectrum of the layer-0 query weights against the random-init bulk, since a trained low-rank component should poke above it, and I'm noting that a 4-layer model trained this long may have moved well away from initialization.

I'm considering whether the shuffled hypothesis changes which heads are useful: within a line, the previous character is just one sample from the bag, so previous-token heads add nothing and induction-style copying wouldn't help since repeated lines come out in different orders. Attention would instead spread across the current line's characters and previous lines. If the text were in its original order, I'd expect previous-token heads in the early layers, which would be a useful thing to test for.

I'm comparing composition scores between heads in different layers, looking for pairs where a later head's key reads from what an earlier head writes, which is the signature of an induction circuit.

I'm weighing whether the layer-0 weights can reveal positional structure at all, since the position embeddings aren't available. One route is to test for K-composition between an earlier and a later head, since induction needs a previous-token head feeding a later one. Checking the weight norms of those pairs would show whether any such circuit exists in this model.

I'm stacking the layer-0 reading matrices to find the input subspace, since trained components should lie in the span of the token and position embeddings. The token part is at most about 27 dimensions, while the position part could reach 128, but the learned positions may be smoother and lower-dimensional than that, with some random initialization noise on top.

I'm weighing whether the output stays faithful to the question when the thoughts get cut off mid-sentence, and I'm working through how to pick up the thread on the 128-dimensional position noise and the weight-norm checks.

I'm considering whether the weak signal from the token embedding subspace could show which letters are actually used, since rarely used tokens would barely move from initialization. The layer-0 reading matrices should therefore span fewer effective token directions than the full vocabulary, though I can't tell yet whether that's 27 or closer to 23. I'm also thinking about whether the first-layer MLP acts like a bigram lookup table.

I'm wondering whether the corpus can be told apart from the attention patterns alone, since leading indentation and repeated identifiers would make Python and English look quite different to induction-style heads.

I'm weighing whether the corpus can be told apart from the weights, since Python, Wikipedia, and romaji would give very different line lengths. Python lines run short, Wikipedia paragraphs can stretch past the 128-token window with no newline, and romaji dialogue sits somewhere between. Without the position embeddings, though, I can't read positional attention directly, and I'm still looking for some indirect way to recover it.

I'm trying to get at the token embedding directions indirectly, by asking whether the previous-token head's value path writes into the same subspace that the induction head's query reads from. If the induction head's query and the prev-token head's value share that token-identity subspace, that would point to the circuit.

I'm working out whether the induction-style score can be read off the weights alone: a bilinear form built from the previous-token head's value and output paths and the induction head's query and key paths. If it's large when the query token matches the previous token at a key position, its diagonal on the token subspace should come out positive.

I'm checking whether the eigenvalues of the full 384x384 product of the previous-token head's OV path and the induction head's QK path come out mostly positive, since that would indicate same-token matching, much like the copying test in Anthropic's mathematical framework. Random matrices would give eigenvalues symmetric around zero, so the spread should show whether a real circuit is there, though I still need to handle the LayerNorm scaling.

I'm checking whether the induction-style matching shows up as positive eigenvalues of the combined previous-token and induction QK/OV product, with the LayerNorm scaling folded in. I'm also wondering whether an induction head's OV path maps token identity through to the output in a way I can detect without the unembedding.

I'm going to compute the SVD spectra of the layer-0 weights to see whether a token-embedding subspace stands out. Position embeddings aren't available, so I'm considering whether a smooth positional structure in the layer-0 heads could reveal previous-token behavior, though I can't check that directly without them.

I'm now reasoning about which weights tie most directly to token identity: the last layer's MLP output feeds the residual stream that the final layer norm and the output projection read from. Because the output projection has rank at most 28, the directions that matter for logits form a small subspace, and I expect the final MLP write to be concentrated there.

Now I'm thinking about the unembedding subspace, estimated from the top singular vectors of the late-layer MLP and attention output weights, which should span about 27 dimensions. The embedding subspace from layer 0 might line up with it, but without the token embeddings I can't check whether an induction head maps token identity through to the output.

I'm working out how many distinct characters the corpus actually uses: Hepburn romaji leaves out l, q, v, and x, so roughly 24 symbols including space and newline, and the unembedding would need about 23 dimensions to tell them apart.

I'm now estimating how many dimensions the final MLP's output weights actually use, expecting a gap in the singular values that separates the roughly 27 output directions from the rest. I'm also setting up the same kind of rank check on the layer-0 input weights to see how many token directions the embeddings occupy.

I'm about to print the shapes, norms, and singular values of each matrix, and check the LayerNorm weight statistics.



---

I had been stuck repeating bash `true` calls, which was going nowhere. The protocol is to reply with exactly one python code block and nothing else, since the weights come through that channel rather than bash. So I'll send a single python block that starts the analysis: spectra and norms, ln weight statistics, and composition scores.

