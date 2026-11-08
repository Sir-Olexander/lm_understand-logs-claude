I'm reading the eigenvalues of head L0H3, and most of them are complex with phases near 45 degrees. Their angles come out at roughly 35 to 61 degrees across the rotation pairs, which fits a positional head that attends to a fixed offset.

I'm considering whether a positional head with varied rotation angles still fits, since the shuffled setting might also call for local heads. I'm thinking through how a line-length signal could be built: an early head would pick out the last newline by recency, and later heads would use that to restrict attention to the current line.

I'm weighing whether a sharp positional pattern fits, since a flat high-rank QK would need many frequencies to peak sharply, and a previous-token head would be a sharp peak at offset -1. I'm also wondering whether the positional embedding subspace could be recovered from the gradients. Then I'm turning to L0H5: its QK is tiny, so it attends roughly uniformly over the prefix, and its OV has about rank two, so it mostly averages a couple of features across the context.

I'm now looking at how the induction circuit could form: a previous-token head writing prior-token information into a subspace, which a later head's keys can read. I'm checking K-composition between head pairs by computing the Frobenius norm of the combined QK and OV matrices, normalized by their individual norms.

Next I'm checking how the query and key matrices of the later heads line up with the value outputs of the earlier ones, using Frobenius norms normalized by each matrix's own size. I'll compute that for every pair, and I'm setting aside the layer-norm scaling between layers for now.

