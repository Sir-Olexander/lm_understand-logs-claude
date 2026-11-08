Most of the symmetric and antisymmetric norms come out roughly equal, which is what a random matrix would give, so it doesn't tell me much since the bulk noise dominates the Frobenius norm. Next I'm looking at the L0H3 eigenvalue phases, where the leading ones cluster between roughly 30° and 60°, with a few outliers near 0°, 84°, and 123°–159°.

The phases cluster around 30 to 85 degrees, which would mean each plane rotates by some frequency times a fixed offset. If the offset is one position, the frequencies would fall in a plausible positional range. I'm wondering whether the attention actually peaks at a nonzero offset, since zero offset would give phases near zero, and these are clearly not zero. That points toward the head attending to something a few positions back, a previous-token or local pattern, though I still need to rule out a token-space explanation.

I'm checking whether the shuffled setup would even let a previous-token head form, since shuffling removes any local order. The normalization runs before shuffling, so the shuffled lines have no local structure to exploit. That would make a previous-token head useless in the shuffled models, which is a strong signal if L0H3 really is one. It also raises the question of where the induction heads come from, since K-composition with L0H3 is only moderately strong.

I'm checking the induction story more carefully by computing the eigenvalues of the composed matrix QK of L3H3 times OV of L0H3, and comparing its magnitude to QK of L3H3 alone. I also want to consider whether layer 3 keys can see L0H3's output directly through the residual stream.

Next I'm looking at the larger L1 to L3 heads, where the complex eigenvalues have phases around 60 to 80 degrees. That pattern suggests they might be working on positional structure too, possibly attending at a fixed offset of one position.

I'm working out whether the attention heads in layers 1 to 3 share a positional subspace, by checking if their complex eigenvectors line up with the ones from L0H3.

I'm going to project L0H3's QK onto the subspace of its top-20 eigenvectors, real and imaginary parts, and compare the fraction of norm kept against a random subspace of the same size. I also want to check whether the top left and right singular subspaces of that QK coincide, since that would confirm a shared positional rotation.

