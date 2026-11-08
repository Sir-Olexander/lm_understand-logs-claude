The two matrices nearly commute, with a commutator far smaller than for random pairs, which suggests they share eigenstructure. That fits with both being functions of a common shift operator or a common symmetric Gram structure. I'll run a simultaneous eigen-analysis: take the eigenvectors of M5 in the d=20 basis, then read off the diagonal of V⁻¹M1V to see how M1 acts on those directions, and compare the angles.

I'm checking whether the angle ratio between the two matrices' eigenvalues stays consistent across the shared eigenvectors. If each head's eigenvalue is a Fourier sum over its positional kernel, that ratio would point to the offsets involved, though I don't know the frequencies, so I'd need another reference head to pin them down.

I'm now thinking about building a shared eigenbasis from a combination of the two matrices, then computing per-head eigenvalues via left and right eigenvectors. The angle pattern across shared directions could reveal which offsets each head attends to.

