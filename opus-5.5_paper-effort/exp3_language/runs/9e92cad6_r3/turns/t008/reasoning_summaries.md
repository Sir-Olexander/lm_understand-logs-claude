I'm working out that head H3's QK reads a shared subspace of about ten dimensions, nearly orthogonal to its OV-input subspace. The 2-d feature that heads 1, 2, 4 and 5 read through OV sits inside the QK subspaces of H3 and H0, but not in their OV inputs, so it may actually be positional.

I'm checking whether H0 and H3 are positional heads, since the 2-d feature also sits in their QK subspaces, and the H1/2/4/5 OV reads might be averaging position or newline-related information. H3's high-rank, mostly positional QK looks more likely than a newline-token explanation, and that would make H0 and H3 look like previous-token or local heads.

Now I'm testing whether H3's QK acts like a rotation: if it's a previous-token head, the bilinear form on the shared subspace should have a strong antisymmetric part with complex eigenvalues whose phase encodes the offset. I'll check that by decomposing the restricted matrix.

I'm going to compute the eigenvalues of the full 384×384 matrix A directly, since its rank is at most 64, and then look at the largest ones by magnitude along with their angles.

