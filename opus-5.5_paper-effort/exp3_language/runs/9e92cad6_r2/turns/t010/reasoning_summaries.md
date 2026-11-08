I'm looking at the first MLP layer's singular values: about 45 are fairly large, with a gap near indices 40 to 50, and I'm wondering whether the residual input is really using that many dimensions.

I'm checking whether the newline prediction could depend on line length, and then I'm wondering why induction heads don't show up here. Layer 0 head 3 looks like a previous-token head, but nothing in the later layers composes with it through keys, only through values, so I'm thinking about a better composition measure that accounts for its high-rank output.

I'm weighing a subspace-based composition measure instead: projecting the key-reading matrix of a later head onto the top output directions of L0H3 and comparing that energy against a random subspace of the same size. The earlier K-composition values sit near the random-rotation baseline, which suggests no special alignment, so I want a check that doesn't depend on the flat spectrum.

