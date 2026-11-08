I'm looking at the overlap between the OV1 and OV5 input subspaces, where about 20 dimensions line up closely. The combined token subspace comes to 48 dimensions, which is more than random overlap would suggest, so the separation isn't clean. I'm considering refining the token subspace by working from the token embeddings directly, since layer normalization seems to be mixing position information in.

I'm planning to find the shared subspace by averaging the two projectors and reading off the eigenvalues near 1, which should give about 20 dimensions. Then I want to split the QK matrices of L0H1 and L0H5 into token and other blocks to see how much of the norm sits in each.

