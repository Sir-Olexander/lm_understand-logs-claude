I'm weighing whether to tell romaji, English, and Python apart by counting the tokens in the set T. The plan is to estimate that token-subspace dimension from the strongly negative same-token quadratic forms in the layer-1 heads.

I'm checking a different way to count the token dimensions: the singular spectrum of the composite U^T OV for L3H1 over the expanded token subspace, which decays gradually across about 29 dimensions. I'm also considering the MLP0 input projection onto T, where a drop after 24 would suggest that many tokens, but since T was built from the top-28 of L0H2's OV input, it may include noise dimensions.

I'm sizing up the positional structure of L0H2's spectrum, since the gradual decline past about 18 dimensions could be carrying position rather than tokens. I'm also thinking about how corpus statistics shape head roles: romaji text repeats short morphemes like "masu" and "desu", so local character n-grams may do most of the work and long-range attention matters less.

I'm working through how head types in each layer could reveal whether the training text is romaji, English, or Python. Local n-gram heads would suggest romaji or English, while long-range copying and induction heads would point toward Python. I'm tallying the head roles layer by layer, with L0 having one previous-token head, one local head, and four broad ones, and L3 having a same-token head, an induction head, and several key-reading heads.

I'm now going through the L1 and L2 heads to label each one, looking at how their key vectors read the b-out component with large weights. The question is whether a head's attention depends on the query at all, or only on the absolute position encoded in the key, which would not give a recency preference relative to the current token.

I'm estimating the size of the query-independent part by computing the norm of the constant query bias times the key's b-out block, and separately the b-out-to-b-out block. The second term is the one that would give a relative, recency-like preference, but its scale depends on the query's b-out magnitude, which I don't know yet. Next I need to check how large the b-out component is in the residual stream at that layer.

I'm checking whether the residual's b-out component could be large enough that LayerNorm at layer 1 would be dominated by it, which seems implausible for a working model. Massive constant-like directions do show up in residual streams, though, so I'll keep that possibility open.

I'm going to pin down which heads in layers 1 and 2 are local and which are broad, and whether their attention depends on the query at all. For each later head I want to check its induction score, its same-token score, its positional rotation, and how strongly its key reads the b-out component, along with the eigenvalues of the b-out-to-b-out block.

