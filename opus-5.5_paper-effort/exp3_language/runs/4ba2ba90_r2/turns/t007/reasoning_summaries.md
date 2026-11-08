Checking the layer-0 heads: with the ln1_w scaling, most entries come out small. L0H2's OV spectrum decays gradually, with about 20 values above 0.3 and about 30 above 0.15, which suggests it carries token-plus-something information. L0H0, L0H1, L0H4 and L0H5 are close to rank 2 or 3, so they likely hold a positional mean or a few token classes such as newline or space.

I'm turning to the later heads now, checking whether the L1 heads act as induction-style copy heads. Their keys may lean on the broad layer-0 heads' positional output, which would make them more positional than content-driven, so I want to measure how much each L1 head's QK reads from that subspace versus the rest.

I'm defining the subspaces to test for induction: the query side from the input space, and the key side from the layer-0 head 2 output that carries previous-token information. I'll form the composite matrix from the L1 query and key weights with that head's OV, then check its top eigenvalues for large positive real values that would indicate matching. I'll repeat that for the later heads and also run a same-token version to see which heads attend to identical tokens.

