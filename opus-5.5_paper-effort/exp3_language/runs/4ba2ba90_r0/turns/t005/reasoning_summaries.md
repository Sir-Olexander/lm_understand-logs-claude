I'm noticing the ln1 weights leave many dimensions small, but the raw spectrum decays smoothly without a clean split at 28 and 128. Next I want to see whether L0H2's large QK component is positional, by checking how much of it sits in the dimensions where the ln1 weights are large.

I'm weighing whether the rank-1 QK in L0H2 works as a positional bias: if the key-side direction grows with position, attention would fall off with distance from the current token, though I'm not sure yet whether that favors the current position or the immediately preceding one.

I'm checking whether a rank-1 QK in this head would just prefer a fixed set of keys, like all spaces or newlines, regardless of the query. That would make it a broad, averaging head over the context rather than a sharply positional one, and I'm not sure the bag-of-tokens reading fits layer 0 here.

I'm now wondering whether a positional embedding could be recovered from the layer-0 query and key weights, since a previous-token head's QK form should give a Toeplitz, relative-position pattern. The token embeddings don't obviously show up in the MLP weights either, so that route looks unpromising for now.

Now I'm looking at whether L1 heads show composition with the L0 previous-token head, scoring all head pairs to find strong outliers. I'm also noting that L1 heads share a negative QK trace around -0.8, which suggests they avoid attending to keys similar to the query, unlike what an induction head would need in residual space.

Checking the composition scores across all head pairs now.

