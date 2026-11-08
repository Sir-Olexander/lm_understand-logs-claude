The enrichment is being driven mostly by the two smallest subspaces, o00 and o04, which each have only two or three dimensions. I suspect these hold the massive dimensions that carry large magnitude after layer norm, so the per-dimension enrichment is skewed toward tiny subspaces with large weights. That makes it a poor signal for finding induction. Separately, the L1 heads' QK behavior seems to be dominated by o04 as well.

I'm checking whether o00 and o04 carry matching position-from-start features between the query and key of the L1 heads, which would make them act more like recency or sink attention than induction. Next I want to drop the tiny special subspaces (o00, o03, o04) and look at what remains in the larger blocks, comparing both enrichment and the absolute fraction of ||B||² in each block.

The L1 heads look like they're mostly driven by position: their QK is dominated by the sink features, and their V reads the L0H5 output strongly, plus L0H1. That's odd if they're meant to be previous-token heads, so I wonder if they broadly average over the window and copy token info instead. Before deciding, I want the V read ratios for T, P, o05, o01, and o for each later head.

I'm checking the V read ratios for T, P, o05, o01, and o02 in each later head, along with the Q and K reads for the same subspaces and the bias-query key preference decomposition.

