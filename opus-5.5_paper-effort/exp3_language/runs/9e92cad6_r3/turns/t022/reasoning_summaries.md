I'm weighing whether the corpus question can be settled by how each model uses its attention heads: copy heads for Python, local and previous-token heads for romaji, with the English case falling between them.

I'm checking whether shuffled Python might show heads tracking newlines, and whether each layer's heads act as broad averagers, content matchers, or sink formers. Next I'm working out how the Fout rotation and the positional part of the L1 heads' QK are contributing, by measuring which subspaces dominate the query and key norms.

I'm now computing the block norms of the attention matrix between the token, H3o, positional, and Fout subspaces, with a focus on whether the H3o-to-T block captures attending to positions whose token matches the previous token, and whether T-to-H3o and H3o-to-H3o blocks show induction-like structure.

