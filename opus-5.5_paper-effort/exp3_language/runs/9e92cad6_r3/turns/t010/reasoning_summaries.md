L3H3 looks like the strongest induction-head candidate so far: with L0H3 as the writer, its trace ratio and norm are the largest at both k=10 and k=25. Its K-composition and Q-composition scores with L0H3 are also the highest in its row, which fits a head that reads the previous token through the key.

I'm checking whether L0H3 really is a previous-token head, since the induction-head reading depends on that. If it instead attends to the current position, the L3H3 match would just be "same character" instead, and the ~45° rotation angles hint at a nonzero offset.

I'm now weighing the broader picture: whether this model looks like a typical character model, where early layers hold positional heads and later layers hold induction heads, and how the shuffled version would differ. Specifically, I'm asking what the rank-2 OV heads in layer 0 might carry, perhaps a line-position feature tied to newlines.

I'm thinking about whether the layer-1 heads could be keying on a line-start feature, but the query side would need that same line-ID to match, and the Q-composition there was only baseline. Maybe the signal is a newline-or-space indicator instead, where a rank-1 QK head with negative eigenvalue would pull a newline query away from newline keys.

I'm now looking for a concrete anchor for the token and position subspaces, starting with the input weights of the layer-0 MLP neurons, since token-detector neurons should load mostly onto the token subspace. I'm also considering whether layer-norm could help isolate those directions, though that seems less direct, and I'm wondering whether the learned position embeddings are simply out of reach here.

Now I'm checking how the layer-1 heads read the 2-d output of L0H1/2/4/5, and whether that feature sits in the positional subspace or in new directions.

Now I'm checking the bias term, which is query-independent, by projecting the key-side query bias through K to see whether it lines up with the output directions of the layer-0 rank-2 heads.

