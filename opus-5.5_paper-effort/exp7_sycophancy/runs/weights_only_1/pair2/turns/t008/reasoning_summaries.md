I'm noticing the weighted maps are dominated by a few massive coordinates, with the read and write directions concentrated on specific dims, so the cosine similarity is near zero. The top singular values in early layers sit about 30 times above the bulk, driven by the norm gain spikes, which makes this view not useful. I suspect the norm weighting itself is distorting things, since the residual stream has varying per-dimension scales from the massive activation dims.

I'm checking whether the norm gains are what skews the weighted maps, since the massive dims may just have small gains to compensate. Without the residual statistics, that weighting looks misleading, so I'm considering working within the head spaces or the gradient structure of the updates instead.

I'm weighing whether the difference lives in the layer-to-layer structure, since the residual activation statistics aren't available to ground the weighted maps. Comparing heads or gradient-direction subspaces might be more productive than relying on the norm gains, which seem to mislead.

I'm wondering whether the difference is better seen through the signs of the updates rather than their magnitudes. If Adam behaves roughly like sign descent, then a rank-1 gradient would leave a sign pattern that is itself rank-1, which might show up as a dominant structure in the difference matrix.

I'm going to check the MLP units next: for each layer, I'll measure how much each unit's gate, up, and down weights changed relative to the layer median, and look for outliers that stand out.

I'm also going to run the same kind of outlier check on the residual-dimension side, looking at the rows of the output matrices.

