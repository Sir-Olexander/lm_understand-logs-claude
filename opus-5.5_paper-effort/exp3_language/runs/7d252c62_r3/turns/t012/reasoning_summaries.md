I'm working out how the first two heads in layer 0 average position: they attend uniformly, and their output carries a smooth 2D positional signal that the layer-1 keys read, along with the query.

I'm checking whether layer-1 heads depend on that smooth positional signal, which would point toward recency-style attention with variable width. I'm also wondering whether these heads are just doing positional work, and considering whether they might instead be attending to everything after the last newline, though that doesn't fit since they depend on average position. Next I want to see whether the magnitude of the average-position term changes what the query picks out.

I'm considering whether the layer-1 heads' reliance on positional structure is actually what the model needs in as-is text, since previous-token heads in layer 0 already cover the local bigram and trigram context. I'm weighing whether the shuffled-text case would call for similar positional copying, and I'm comparing induction-style query and key signals in layers 2 and 3, which look only modestly enriched.

Now I'm checking the layer-3 head with a symmetric query-key match at about 0.9, where both sides read the same features, which could mean it attends to positions sharing my context. The earlier induction-style signals are only modest here. Still, the rotation from the MLP in layer 0 might mean the token features don't line up the way I'd expect, so I'm asking what kind of same-context matching would actually help.

I'm weighing whether a symmetric same-features match could be useful at all, since copying the matched token would just return the current one. The asymmetric option, where the query reads the current token and the key reads the previous-token features, looks more plausible. Shuffled text could still favor a head that counts how often the current character already appeared in the line, to track what remains.

I'm trying to pin down the positional embedding structure by treating the positional subspace as a sum of sinusoidal components, so the query-key matrix would rotate each frequency plane. I'm looking at what the eigenvalues of that matrix would reveal about the positional frequencies.

I'm estimating the eigenvalue angles in L0H1 and L0H5 and asking whether an offset-1 head with frequency ω would produce a rotation of that size. The ~55° cluster in L0H5 doesn't match a simple doubling of L0H1's frequencies, so I'm checking whether the two heads use different positional frequencies rather than the same ones.

I'm thinking about whether the positional eigenvalue angles could be reconstructed from the sinusoidal assumption, and whether the ~55° cluster in L0H5 would fit a simpler frequency relationship. I'm stuck going in circles on that, so I'm stepping back to consider which head patterns are typical for this kind of setup.

I'm weighing whether a sharp offset-1 match or a broader recency window fits better here, since a local window would roughly approximate the current line's characters in shuffled text. Distinguishing the two depends on how sharp the positional query-key pattern is, and I'm stuck on how to read that from the embeddings alone without knowing their scale.

I'm checking whether the 180° eigenvalue in L0H1 is just the period-2 alternating component, since a rotation of exactly 180° would mean the offset-1 reading only holds if that frequency is the one doing the work.

I'm looking at the L0H5 eigenvalue angles of roughly 53 to 57 degrees, along with a 29-degree component and a real-positive part, and wondering whether that real part is a self-like term. I'm also considering how L0H4's running-average output could trace a monotonic curve over positions that layer-1 heads read through their keys.

I'm asking whether layer-1 heads could read the averaged-position subspace jointly with the query's own position, which would make the preference relative to the current position. Since the token at the query sets how strongly positional preference applies, the pattern might shift by token type, such as after a space versus after a letter.

I'm going to test the query and key sides using the layer-0 averaged-position output, along with the other position and token signals, to see which layer-1 heads pick up relative positional comparisons.

