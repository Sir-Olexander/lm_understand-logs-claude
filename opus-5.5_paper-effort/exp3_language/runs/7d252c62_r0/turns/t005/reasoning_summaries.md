I'm looking at the layer-norm weights and the attention weight matrices, and the large ln1_w dimensions line up with the key weights for two heads while anti-correlating with the value weights. That makes me suspect the residual stream splits into positional coordinates and token coordinates, with the embedding dimensions partly separated.

I'm checking whether the residual stream's coordinates split into positional and token groups, so I'm computing effective column norms for the key and value weights per head. The next step is to test whether the principal subspaces from those weights are actually distinct, using the scaled matrices.

I'm working out how the previous-token behavior of L0H1 and L0H5 might show up in a Fourier-like positional embedding, where the eigenvalue angles in each frequency plane would have to match the frequencies. The 58° pair suggests a period around six positions, and the 129° and 135° pairs suggest something closer to 2.7, which are high frequencies.

I'm reasoning about how the frequency of the positional embedding relates to the eigenvalue angle. For a previous-token head, the angle should match the frequency itself, so the 180° component fits a period-two pattern that favors odd offsets, including the one-step lag. I don't know the frequency yet, though, so I can't pin down the peak offset directly.

I'm checking whether the 180° eigenvalues reflect a period-two positional component favoring odd offsets, which would fit L0H1 and L0H3. The token subspace could also produce negative real eigenvalues, so I need the positional subspace separated before I can decide between the two.

I'm wondering whether the L1 heads, which show angles around 73°, are also reading positional structure, or whether that's a different mechanism altogether.

I'm weighing whether the L0 heads would do something different on shuffled data, since a prev-token head would be pointless there. The heads that attend uniformly or to the most recent newline would still make sense, which could help tell the as-is and shuffled cases apart.

I'm thinking about whether the L0 heads' attention is driven by positional structure at offset one, which would need high-frequency components in the positional embedding. I'm also considering the token subspace: with only 28 tokens, its contribution to the OV circuit should be low-rank.

I'm going over the OV spectra of the L0 heads to count how many singular values stay well above the noise floor, expecting roughly 27 if the value path only reads token information. I'm also planning to pull the token subspace from the L0 OV right singular vectors and the positional subspace from the positional heads' QK, so the two can be compared.

I'm now checking whether L0H4's uniform attention, which reads only two dimensions, falls inside the token subspace or the positional one. The K-composition scores from L1 and L2 keys onto the L0 prev-token heads sit near baseline, which seems to argue against a standard induction circuit, though the large L0H4 and L0H0 contributions to the key weight norms could be masking the signal. I want to project out those output directions first and then recompute.

I'm checking whether the L0 OV spectra are rank-limited near 27, which would mean the value path mostly reads token information. I'm also thinking about projecting the P_T-restricted composition trace of the L2 and L3 keys onto those prev-token outputs, since induction might come through L1 rather than directly.

I'm checking whether the L2 keys read from the output subspaces of the earlier heads, and whether the K-composition is low only because the big L0H4 and L0H0 contributions are swamping the signal. To make it independent of OV magnitude, I'm measuring the key weights' norm along each head's top left singular directions and comparing that against a random orthonormal baseline of the same dimension.

I'm going to count the singular values in each layer-0 OV matrix that stay above the noise floor, and see whether the value path only carries about 27 token-related directions. I also want to measure how much the key weights read from each earlier head's output subspace, compared against a random orthonormal baseline of the same size.

