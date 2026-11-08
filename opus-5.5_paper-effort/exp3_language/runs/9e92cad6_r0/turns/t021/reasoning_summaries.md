I'm looking at W_fc.0 with the top 60 attention output directions projected out, and the spectrum has a knee between indices 25 and 26. Counting the dimensions above that knee gives 26 at each of the three ranks I'm checking, r=40, 60, and 80.

I'm checking whether this 26-dimensional subspace comes from token embeddings or positional information, since positions also feed into MLP0. I'll compare these directions against the positional subspace from the L0H3 QK circuit and also look at how sharp the knee is after the gradual decline.

I'm now planning to compare the 26 directions against the positional subspace P and the L0H3 OV-input directions T, to see whether they're mostly orthogonal to position. If they are, that would point to a token subspace of about 26 dimensions, which would fit a larger active vocabulary like English or Python rather than romaji.

A 26-dimensional count would fit a larger token set, but the 24 tokens of romaji could still give that number if a few positional dimensions are mixed in. The next step is to measure how much of this subspace overlaps with the positional directions from the L0H3 QK circuit.

