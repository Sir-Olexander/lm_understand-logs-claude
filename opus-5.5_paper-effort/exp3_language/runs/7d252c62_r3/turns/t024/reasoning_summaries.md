I'm noting that L0H1 and L0H5 share a key-token direction with 0.91 overlap, and H0 looks related too. The shared token might be a space or a newline, which would suggest positional heads that adjust based on the key token. I'm also weighing whether a more global inference approach would be cleaner, and I'm considering whether the positional embeddings can be recovered from L0 to test previous-token versus local or broad attention.

I'm working out how the learned part of the query, key, and value weights for the first layer lies in a subspace tied to the embedding span, since the gradient updates are built from normalized inputs. I'm checking whether that subspace can be estimated, which would separate token directions from position directions, though it wouldn't recover the individual position embeddings.

Positional attention is the next thing to pin down: if the position embeddings are low-rank and smooth, the head is probably broad, but a sharp previous-token head would need higher-frequency components. I'm also looking at whether a previous-token head should avoid attending to itself, so I'm comparing a query's self-score against its score for the prior position.

I'm checking the symmetric parts of M1 and M5 to see which heads favor self-attention. M5's is mostly positive, so a local or self-including head is likely, while M1's is indefinite, leaving its pattern unclear.

I'm checking whether M1's pattern can be read from its symmetric part, since M5's antisymmetric part is large enough to shift its peak away from self. Without the positional embeddings I can't settle either head, so I'm looking at whether L0H1's 180° component says anything about its sign.

I'm checking whether L0H4's low-frequency input directions line up with the 180° eigenvector of M1, since that would tell me if the negative eigenvalue is an alternating, previous-token signal rather than a smooth effect.

I'm checking whether L0H4's low-frequency input directions line up with each joint eigenplane, which would confirm that the 180° component is smooth rather than alternating.

