I'm checking the random-init norms first: a 64x384 weight with std 0.02 comes to a Frobenius norm of about 3.14. Layer 0's heads 1, 2, 4 and 5 sit below that at around 4, so the first thing to note is that head 5's QK is close to random and has decayed, while its OV is large. Head 3 looks like the opposite, with a big, high-rank QK (about 32 significant dimensions), and head 0 is also large.

Layer 1's heads all show the same key pattern, with two dominant singular values, which might be a positional signal or newline detection. Layer 3's output projection has roughly 28 large singular values, so the final layer seems to write into a subspace of about that size. The open question is whether the data is shuffled, and how to separate positional from token-based query-key structure in layer 0.

I'm reasoning about where the layer-0 input comes from: the residual stream is LN of token plus positional embedding, and the token part should occupy roughly 28 dimensions. The positional part is harder, since a learned positional embedding in a 128-context model tends to be smooth, and a previous-token head would show up as a query-key peak one step back. Recovering that positional component directly from the weights doesn't look straightforward.

Still working out how to separate the token and positional parts of the layer-0 input. Layer 1's K-composition with a layer-0 previous-token head may be the more tractable route.

I'm now weighing how a shuffled-data model would need to track the current line: a bag of characters plus the distance since the last newline. A uniform head over the current line can't stop at a newline under causal softmax unless earlier layers mark how many newlines have appeared, so that's the feature I'd look for.

I'm planning to analyze the layer-0 inputs by stacking the query, key, value, and MLP readers scaled by their LayerNorm weights, to see which input directions they actually use. Head 3's large, high-rank query-key pattern might be the positional previous-token head, so I want to check that against the subspace.

I'm going to build a Gram matrix from the layer-0 reader weights, scaled by their LayerNorm gains, and look at its eigenvalue spectrum to see how many input directions carry real signal.

I'm checking how the spectrum of W_fc.0 compares to a random bulk: with std 0.02 on a 1536x384 matrix, the random singular values should fall between about 0.39 and 1.18. The value at index 200 is 0.925, which sits inside that range, so I want to see the full spectrum to find where the learned part meets the bulk edge. That crossing should indicate how many input dimensions the layer actually uses. I'll also look at the large singular values in W_proj.3.

I'm going to count how many singular values stay large in the final layer's output projection, then print the full spectra to see where the learned part of the layer-0 reader matrices meets the random bulk. I also want to check whether the residual stream directions that layer 0 reads are only the embedding subspace, and whether absolute position still matters for prev-token heads.

