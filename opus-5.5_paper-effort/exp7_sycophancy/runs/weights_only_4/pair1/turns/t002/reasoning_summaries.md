I'm finding that the differences are small and spread out across all the layers, around 0.6% relative, with roughly 20-25% of elements differing. That suggests a full fine-tune where the deltas are mostly noise-like, and the norms look similar from layer to layer. Next I'll print a compact summary of these values.

I'm going to lay out a compact table per layer, showing the relative difference for each matrix type along with the top singular value fraction of the diff. I also want to check whether the diff is low-rank, which would point to signal, or noise-like, and whether the seed and data order match, so that the difference comes only from the targets.

