I'm checking that the 26 dimensions FC0 relies on are token-driven rather than positional, and they appear to be. Since the input is token plus position, these 26 dimensions must sit within the token embedding span, apart from any positional components the query-key read doesn't pick up.

I'm now weighing whether a dimension that carries only rare tokens could be missed, since the layer-0 value heads that read the token embedding may simply ignore tokens like q and z. Checking the remaining FC0 dimensions beyond the 26 would need a criterion that doesn't rely on the positional signal, and so far the candidates around dimensions 26 to 29 look like noise.

I'm checking whether dims 26–29 of FC0 could be residual attention-output components or rare-token readers rather than positional signal. Their non-positional enrichment and singular values suggest they're learned, so I'll test their alignment against the full layer-0 OV output spaces. If they fall outside both attention outputs and the positional subspace, they'd be candidates for rare-token dimensions.

