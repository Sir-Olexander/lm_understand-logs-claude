I'm checking how MLP0's output lines up with L0H4's contribution to the o0.4 plane, since both seem to write into that same subspace.

I'm checking whether the plane MLP0 writes into could also be tracking something like newlines, since the positional heads might key on line boundaries and I haven't accounted for that yet.

I'm working out the total input space of layer 0 by stacking the input weights for the query, key, value and MLP projections and checking their singular values. That should show how many dimensions are really used, since the positional part may be fairly low-dimensional. I'm also considering whether the token directions copied by the first-layer heads can be separated cleanly from the position directions.

I'm checking whether L3H5's query and key read the same subspace, which would make it behave like a same-feature matcher in the last layer. Its value reads the token directions strongly, so if it attends to matching positions it would copy the token there. For induction, though, the key at position j+1 would need to encode the preceding context, and I'm not sure the symmetric QK structure supports that yet.

I'm working through how a head's key at position j would need to carry the context from position j-1 for induction, and whether a symmetric QK match on a shared subspace can support that.

I'm now comparing L3H5's key-side input subspaces against the outputs of the layer-2 heads, to see which of them feed the space it reads on the key side.

I'm thinking about whether the token subspace could be recovered from the first MLP's weights, though that's unclear. I'm also considering the corpus itself: the positional and line-length structure could reveal what kind of text this is, since Python lines are short and the window is 128 characters.

Thinking through how Python's indentation differs from English prose here: leading spaces get kept as runs, so a line-start has to be counted and matched against the previous line's indent, while English paragraphs run so long that a 128-character window rarely crosses a newline.

I'm now looking at whether the uniform heads in layer 0 copy token identity or only position, since that changes what a broad bag-of-characters strategy would need. The heads that look like they'd do the copying seem to be the ones attending across the whole window.

I'm checking whether L1's heads copy token identity, since they read the token directions and the L0H5 output, and whether they attend broadly across positions.

I'm estimating how sharp or broad the L1 heads' attention is by looking at the score magnitude, since the smooth 2D position curve would make attention either spread out or concentrate at the edges of the context. I'm also second-guessing whether I've left out the 1/8 scaling in the QK matrix, which would change the size of the scores.

I'm weighing whether the L1 heads' attention would be sharp or broad, but the scale of the residual stream is unclear, so I can't pin that down yet. The layer-0 heads are where the decisive structure should be, so I'm going to try recovering the positional kernel of L0H1 and L0H5 by assuming a shift structure: a single orthogonal rotation that maps each position to the next and commutes with both heads' matrices.

I'm working out the rotation R that would shift each position to the next, and assuming it acts on each joint plane with an unknown frequency ω_k. If head H1 is a sharp previous-token head, its eigenphases would pin each ω_k directly, which would give the attention score as a sum of cosines over the plane frequencies.

I'm now testing the hypothesis that H1 is a sharp previous-token head by setting each frequency to its eigenphase, then checking whether H5's resulting kernel over offsets 0 to 20 peaks at a sensible offset. Alternative offsets for H1 would make the frequencies ambiguous, so that case needs its own check.

I'm assuming the frequency of each rotation plane is set by the phase of H1's eigenvalue, so I can compute H5's kernel over offsets and check whether it peaks somewhere sensible. The planes where H1 is weak give unreliable phases, so those frequencies may be off.

