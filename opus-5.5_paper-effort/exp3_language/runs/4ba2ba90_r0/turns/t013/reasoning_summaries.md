The L1 keys look like they're reading the 3-dimensional output subspace of heads L0H0, L0H1, L0H4 and L0H5, with roughly 63 to 76 times enrichment. That would put about 55% of W_K's energy into those three dimensions, so dimension 222 and its neighbors seem to dominate that channel. L2 keys show a weaker version of the same pattern, and L3 a smaller amount.

L1 values seem to carry the L0H2 output, probably previous-token identity along with token identity. The open question is what the channel-222 feature actually encodes. If L1 queries also read it, the attention could be matching positions, such as which line or word a token belongs to, so that each head aggregates within the current unit. I still need to work out how L0's rank-1 QK scores would produce that feature.

I'm working through how a head with uniform causal attention could produce a newline count, which would let later layers pick out positions in the same line. Matching on the number of newlines seen so far would tend to pull in just the current line's tokens.

The idea floated here is that a head with uniform causal attention could track how many newlines have appeared so far, and a recency-weighted newline score could also pick out positions in the current line. Keeping that in mind, I'm wondering whether channel 222 responds to a space-like or newline-like signal, though I don't have labels to check that yet.

I'm checking whether the L0 heads' key vectors pick out a specific class of tokens, but I'd need embeddings to confirm that. A different angle is the layer-norm statistics, since token embedding norms might separate newlines from spaces. The more direct test is to project each L1 head's QK onto the three-dimensional L0 output subspace on the key side and see what the query side looks like.

I'm checking how the L1 queries read the channel: the query side of each head is probably dominated by its constant bias term, which would make attention simply favor keys with the largest value of that feature. To test that, I'll compute the constant query contribution for each L1 head and see how it projects onto the keys that read the L0 output subspace.

I'm restricting the key side of each L1 head to the three-dimensional subspace S and looking at what the query side reads from it. I also want to check whether the query's bias term gives the keys a fixed preference, by computing that key-side bias vector and seeing how much it overlaps with S.

