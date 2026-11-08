I'm looking at which heads have their query-key matching mostly in the positional subspace, with L0H1 and L0H5 looking like the classic positional ones that copy token content. For L0H0, H2 and H3, the key preference seems tied to tokens, while their value outputs draw mostly from positional information, so L0H0 and L0H4 may be averaging positional signal.

I'm now considering the layer-0 head L0H4, which seems to average positional embeddings over prior positions, giving a smooth absolute-position signal that layer-1 heads could use for recency or local attention. I'm also wondering whether the LayerNorm scaling, which varies by token and position, complicates that picture. Then I'm turning to L0H1 and L0H5 as positional-query-key heads that copy token content, and asking whether the model would learn such positional heads on shuffled text.

I'm now asking whether these positional heads could be learned on shuffled text, where recency-weighted averaging over nearby characters would still give a useful bag-of-context signal. The key question is what attention offset the positional query-key score encodes, and whether the positional embeddings could be recovered from the structure of that score matrix.

I'm working out what a previous-token head would need from its query-key matrix: the score for attending to the token just before should beat the self-score, which means the symmetric part should be low on the positional subspace. For the L0H1 eigenvalues, the negative real fraction suggests self-attention is being suppressed.

I'm weighing whether the positional embeddings could be recovered from the structure of these QK matrices, since each eigenvalue angle would correspond to a frequency component. Without knowing those frequencies, though, that doesn't resolve cleanly, so I'm looking for another route to the residual stream.

I'm trying to see whether the positional embeddings could be recovered from the learned weights, given that W_q.0's input-side updates should span roughly the same directions as the normalized token-plus-position embeddings, which makes the two hard to separate. I'm also weighing whether the MLP0 input weights offer a cleaner route, and then I'm turning to how the layer-2 and layer-3 heads K-compose with the layer-0 previous-token heads L0H1 and L0H5, with composition scores around 0.05 to 0.08.

I'm now checking the later heads for composition with the previous-token heads, since the scores I have so far are diluted by Frobenius norms. The plan is a more targeted measure: restrict the query side of each later head to the token subspace and the key side to the output of L0H1 or L0H5, then compare those bilinear forms for induction-style structure.

I'm setting up a targeted measure for the later heads: restricting each query side to the token subspace and each key side to the output of L0H1 or L0H5, then checking the bilinear norm against a random-subspace baseline of the same dimensions.

I'm defining a measure of how strongly each later head's query side, restricted to the token subspace, lines up with the key side from L0H1, L0H5, L0H4 or L0H0. I'll normalize it against the dimension-matched random-subspace baseline to see whether any of these pairs stand out.

