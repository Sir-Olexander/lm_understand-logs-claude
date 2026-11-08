I'm looking at L0H2's output side, where the singular values stay fairly flat across roughly 20-25 dimensions before dropping off, which fits a head copying token identity given about 27 effective tokens. Its QK pattern is large and high-rank, pointing to positional behavior, so it's probably a previous-token or sharp local head.

I'm weighing whether a head that copies token identity could be useful in shuffled text, since attending to one fixed relative position would mostly be wasted. Then I'm realizing that the previous character still carries information about the line's overall character distribution, so a sharp positional copy might not be useless after all.

I'm checking whether L0H2's attention is sharply positional or content-based by comparing where its QK weights live against the token dimensions its OV reads. The L1 heads only pick up its outputs through value composition, not through query or key composition, so that path is more about carrying previous-token information forward than about induction.

I'm checking whether L1's key weights line up with L0H2's output subspace, using the top ~20 singular directions and comparing against the random expectation of 20/384.

