The simultaneous diagonalization attempt fell short, with complex eigenvalues and poor matching, so the forms don't look diagonal enough. I'm considering a different route to recover the token embeddings, but the idea of using a norm after layer norm doesn't pan out. Switching to the corpus side, I'm looking at how positions are used statistically, starting with L0H5 as a previous-token head.

I'm weighing whether some heads in layer 0 respond to specific kinds of tokens, like newlines or vowels, rather than one fixed token. L0H2's token preference looks like it could be a graded direction across a class of tokens instead of a single one, which would explain why its copying reads as broad.

I'm checking whether the MLP0 neurons cluster around token-specific directions, since a detector for one token would have weights roughly aligned with that token's dual embedding. I'll project the normalized W_fc rows onto the token space and look at the Gram structure for clusters.

I'm thinking the token-embedding recovery through the layer-norm route isn't useful, so I'll look at the corpus side instead. The current guess is English or Python, since the token count points to a small character set. Python looks more likely to me because its newlines and indentation are frequent and its repetition should drive strong induction behavior, whereas English would show that less.

I'm looking at L3H3 next, which has a large OV circuit but a small QK one, so it likely acts as a broad averaging head writing a strong low-rank signal. L0H4 seems to do something similar. I'm also wondering about the anti-self behavior in L1, where same-token attention is suppressed, and whether a local head avoiding repeated spaces in indentation would look different from one avoiding repeated letters in English.

I'm checking whether the anti-self pattern in L1 is uniform across all token directions or concentrated in a few. If one direction dominates, like spaces, it would mean the head avoids repeating that token specifically, which could matter for indentation. I'll look at the eigenvalues of the symmetric part of the token-to-token QK block for each L1 head.

