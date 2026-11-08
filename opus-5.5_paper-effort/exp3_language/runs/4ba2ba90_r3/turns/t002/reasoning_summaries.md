Now I'm looking at the real data, and it contradicts what I thought I saw before, so those earlier observations were hallucinated and I'm starting fresh. The layer-0 attention query-key norms are very small for several heads, which would make the scores close to constant and the attention roughly uniform over all previous positions.

Now I'm checking the attention scale more carefully: the layer-0 heads with small QK norms would still be near-uniform even allowing for the layer norm weights, since a random-init baseline gives a similar magnitude. The bias terms are small too, so those heads most likely average over all previous positions, while the head with the large norm is the only one that could attend sharply.

Scores here are still in a plausible range, so these heads aren't necessarily uniform: with the layer norm scaling already folded into the query and key projections, the attention matrix's top singular value could push some logits to a few units. That makes the earlier "essentially uniform" call for heads 0, 1, 4 and 5 look too strong. Head 2, with its norm of 32.6, is the one I expect to attend sharply, and I'm now looking at where it lands.

Now I'm reading the layer-1 heads: their query-key ratios come out negative across the board, and their OV traces are strongly negative too, so they may be suppressing whatever they attend to. The layer-1 key biases are large, which is worth noting. Layer 3 heads 0 and 1 show clearly positive query-key ratios, pointing to same-token matching, and their OV values are large.

Now I'm working through the L0 heads and the L3 matching: the L1 heads pick up the four near-uniform L0 heads with weak positive ratios, and L3H1 draws on several earlier heads. The broad L0 heads average over every previous position, while L0H2 stays sharp, and I'm wondering whether the averaging is what matters most when the text is shuffled.

Now I'm asking whether L0H2 is sharp because it tracks content, like newlines, or position, and I'm thinking about a head that attends to the last newline to find where the current line starts, with its output carrying mostly position information and so a small OV norm.

Now I'm thinking about whether L0H2's sharp attention comes from a rank-one gate on a specific token, like the newline, or from a positional rotation. A single dominant singular value in its attention matrix would point to the token-gating case.

Next I'll compute the singular values of A for each layer-0 head and for L3H0 and L3H1. I'm also wondering whether the broad layer-0 heads still have some structure despite their small QK norms, possibly a rank-one pattern keyed on a particular token. A bound on the scores from the layer-norm-scaled inputs, which have norm about 19.6, would tell me how large the logits can get.

Now I'm estimating how sharply each layer-0 head can attend. L0H2, with its large norm, could produce scores in the hundreds, while L0H1's are at most around one, so it stays nearly uniform. The broad heads L0H0, H1, H4 and H5 are likely averaging over everything from the start of the window up to the current position, not just the current line.

Now I'm weighing whether uniform averaging in layer 0 makes sense for natural text, since I'd normally expect local prev-token heads there, and four near-uniform heads out of six looks more like what shuffled data would produce.

Now I'm weighing whether the key biases in layer 1 matter, since a large query bias dotted with the key content would give a query-independent preference across positions.

Now I'm wondering whether the key bias term matters at all, since it adds the same value to every score for a given query and so drops out of the softmax, leaving only the query bias dotted with the key content to set a preference across positions.

Now I'm weighing whether the key bias even matters, since it adds the same amount to every score for a given query and drops out of the softmax; what should set the preference across positions is the query bias dotted with the key content.

Now I'm thinking about how to separate token-driven structure from positional structure in these heads. The broad layer-0 heads seem to read token identity, so their OV input directions could mark a token subspace. I want to check whether L0H2's query-key pattern lies within that subspace or outside it, which would tell me if it's positional.

Next I'm planning the spectral checks: taking the singular values of each layer-0 head's attention matrix and its OV circuit, and then testing whether the broad heads' input subspaces overlap with the query-key subspace of L0H2 and L0H3.

