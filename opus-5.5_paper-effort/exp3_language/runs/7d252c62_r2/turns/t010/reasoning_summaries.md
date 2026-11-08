The layer 2 and 3 enrichment stays close to 1 for most features, probably because their residual stream is mostly filled by other outputs, especially MLP outputs. Those heads' QK circuits may also read MLP outputs, which I haven't accounted for yet. MLP0 and MLP1 could be carrying the previous-token information, since MLP0 seems to compute bigram features from the o05 output, so the induction keys might be drawing on MLP outputs too.

Now I'm checking the layer 1 heads: their values draw on the copied token from layer 0 and the current token, their queries on the token and the previous character, and their keys mostly on the sink position and the token, with little on the previous token. I'm also weighing whether the shuffled-line case needs a line-id signal, which would mean a layer 0 head copying positional information from the last newline.

I'm checking whether L0H0 could be the line-start mechanism: its query reads position, its key partly reads the newline token, and its value copies position, so it would find the most recent newline and carry its location. L0H3 looks similar. If L1 heads then match line-start information on both sides, they would attend within the same line, which would explain the shuffled-text behaviour.

I'm working through how the L1 heads might track line starts: L0H0 and L0H4 both write positional summaries into the same subspace, which could let L1 attend within a single line. That would explain the shuffled-text behaviour, though I still need to find where the induction mechanism sits, possibly in L2/L3 through MLP-derived prev-token features.

I'm now considering a composed-circuit test for induction: the query reads the current token, the key reads the previous token via L0H5's OV path, and I'd compare that matrix against the direct same-token matrix. Since I only have the subspace rather than a token basis, I'd check positive-definiteness of the eigenvalues, which doesn't depend on basis.

I'm now sketching the token-space circuits by head, starting with the same-token pathway for each layer 1 and later head, then checking whether its symmetric part is mostly positive.

Now I'm defining the same-token circuit for each head and comparing it against the induction-style paths that route through L0H5, L0H1, and L0H2, including a two-hop version for the later layers.

I'm comparing the candidate circuits on a relative scale, so the magnitudes can be set against each other.

