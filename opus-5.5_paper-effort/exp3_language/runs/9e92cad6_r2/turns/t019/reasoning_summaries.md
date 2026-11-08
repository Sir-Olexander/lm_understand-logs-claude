The L3H3 query-key block for the transposed-to-shared pairing is the strongest one, with a larger magnitude than random and high positivity. Its random-subspace counterpart is also mostly positive, which suggests the whole QK matrix is close to positive semidefinite, so the shared-subspace pair looks much like random-random. I'm about to compare the qT side against that.

I'm checking whether the strong positive L3H3 cross term could just be a side effect of the overlap between the T and S subspaces. If QK were roughly a positive multiple of the identity, the trace of T^T S would come out negative given OV3's structure, but I'm seeing +0.476, so this pairing looks like a genuine positive match rather than a generic artifact.

I'm reading L3H3's attention as a possible induction-style head: if L0H3 looks back one position, L3H3 would copy the token that followed an earlier occurrence of the current token. I can't confirm the output side without the unembedding, since its OV trace is negative and the direction is unclear. I'm also wondering whether shuffled text, with its repeated fragments like "else:", would give rise to a similar pattern.

I'm considering whether the same reasoning holds for shuffled text: within a line, the token after a repeated character is just another random character, so there's no induction signal to exploit, and the head would only be picking out same-line positions. That makes it a weak explanation, which leaves me wondering why the L1 and L2 heads look as homogeneous as they do.

I'm checking whether L1 and L2 act as local context gatherers, attending to recent positions with a recency-weighted pattern, while L0H3 alone supplies the sharp ordering signal. That would make the as-is model a crude but workable n-gram-like setup, and I'm wondering whether the shuffled case builds a similar same-line bag instead.

I'm now checking whether the L1 and L2 outputs are a recency-weighted sum over key positions, which would give a monotone increasing weight as the key moves later in the sequence, or a step-like signal keyed to newline content. The L0 group's query-key direction reads token identity strongly, so the weighting may depend on which tokens are present rather than on position alone. Without the embeddings I can't tell which tokens drive it, so I'm weighing a multi-scale explanation: four L0 heads with nearly identical 2D input-output maps but different strengths, each contributing a scaled copy of the same direction.

I'm going to compute, for each group head, the 2x2 map from its inputs to outputs, and then the key-only effect on the L1 logits. If the group heads push with opposite signs, the logit becomes a difference between two weighted running means, which might act as a line-level feature.

