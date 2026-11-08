I'm noting that every layer 1 and 2 head, plus a few in layer 3, carries a large key-preference vector that comes from the query bias and doesn't depend on the query. Nearly all of that vector sits in the 2-d subspace that the layer 0 heads write to, so these heads seem to favor keys where that 2-d feature points strongly in one direction.

I'm now checking whether the feature these L0 heads compute is positional, perhaps a running average of position embeddings, which would give later heads a recency preference. I'm also wondering why the model would route this through L0 heads rather than using a direct positional signal.

I'm now considering whether the first head's key preference might instead encode the position of the most recent newline, so keys in the current line would score highest. That would let the model focus on the current line's characters, which a shuffled-text task would need. I'm also comparing this with the uniform-attention head, whose query direction differs from the others.

I'm checking whether the 2-d feature these L0 heads write is in the token subspace, since the overlap with the token direction looked small. The overlap with L0H3's query-key subspace was large, which points toward a positional signal. I want to see whether the key direction of L0H1, H2 and H4 lines up with the token subspace, which would mean they attend to one specific token and copy its position.

I'm checking whether the key directions of L0H1, H2, and H4 line up with the token subspace, since that would mean they attend to one specific token type and copy its position. I'm also going to look at their query directions and the query-independent key preferences that come from the L0 bias.

