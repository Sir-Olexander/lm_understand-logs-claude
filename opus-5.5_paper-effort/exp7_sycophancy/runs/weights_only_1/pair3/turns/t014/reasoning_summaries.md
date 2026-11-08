Late layers 14 to 17 show strong overlaps among their writers, well above chance, which points to a shared output subspace. That fits answer-token directions or a common gradient direction. Next I'll build a consensus subspace from the top vectors of layers 15 to 17 and the output projection, then run the copy test on each head.

I'm building the consensus subspace by stacking the layer 14 to 17 subspaces into 128 vectors and taking the top singular vectors. For each head I'd compute a projected overlap score for both conditions, take their difference, and compare it to random 16-dimensional subspaces to get a z-score. The same e-coordinate test should also work since the source token identity is already in that space.

