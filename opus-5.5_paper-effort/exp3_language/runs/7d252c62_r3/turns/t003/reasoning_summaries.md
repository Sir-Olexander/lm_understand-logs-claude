I'm looking at the layer-0 LayerNorm weights, where most values sit near 0.1 but roughly 80 dimensions are much larger, and I suspect those large dimensions are the ones the embeddings feed into attention. To test that, I'm going to look for the subspace the token and position embeddings occupy, starting with the input-side singular vectors of the combined weight matrices.

I'm working out which per-head QK eigenvalues to check, computing the 64x64 eigendecomposition for each layer-0 head and looking at the phases, and I'll also account for the LayerNorm bias contribution to the queries before checking all layers.

