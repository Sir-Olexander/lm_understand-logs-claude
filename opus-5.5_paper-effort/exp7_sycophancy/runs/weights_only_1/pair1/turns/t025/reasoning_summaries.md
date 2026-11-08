I'm checking whether the overlap between the T-like subspace and the Δq read directions could be inflated, since D_fin weights certain dimensions heavily. To test that, I'm setting up a null baseline using random 5-dimensional subspaces multiplied by dfin and measuring their overlap with v.

I'm checking whether the late-layer Δq read directions aligning with T*dfin (0.23–0.52) could simply be an artifact of dfin concentrating weight on a few dimensions, so I'm building a null baseline from random 5-dimensional subspaces scaled by dfin and comparing their overlap with v. I also need to sort out which "prediction direction" is the right one: the residual direction D_fin⊙e versus the writer direction T/D_fin, since the reader at L0–2 and the late read vector live in different spaces.

