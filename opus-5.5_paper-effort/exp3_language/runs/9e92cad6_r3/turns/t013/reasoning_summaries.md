I'm looking at the L0H3 OV spectrum: the first 18 values sit around 0.52 to 1.13, then there's a drop at index 18, which would suggest about 18 strong dimensions followed by a tail. That's not a sharp enough cutoff to pin down a token count, though, so I'm weighing whether the "Fin" two-dimensional input subspace might instead reflect newline or space token structure.

I'm weighing whether the Fin subspace overlaps with the positional direction, since it matches it closely (0.91-0.97). If it does, H5 might be using positional features on the query side against token features on the key side, so its attention would depend on where it is in the sequence. I'm also considering that P/Fin may instead be a newline/space direction that LayerNorm amplifies.

I'm now going to check whether the later heads' query and key sides line up with the T, P, Fout, and H3out subspaces, by measuring how much of each QK block falls into each candidate basis.

I'm noting that the subspaces aren't orthogonal, so the overlap numbers will only be a rough guide.

