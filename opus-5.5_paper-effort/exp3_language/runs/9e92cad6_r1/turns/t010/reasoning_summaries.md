I'm reading the layer-1 heads and noticing that Q, K, and V all draw heavily on the token signal, with the previous-token info also strong in Q and V. Raw position barely registers, which seems odd. I'm also wondering whether P, defined from QK3-in, is picking up some token mixing, given the 0.52 overlap between T and P.

I'm going through the remaining layer-1 heads and noticing they look alike: Q leans on O3 and O5, while K draws on T, O3, and MLP0, which makes them read as content-focused heads. Layers 2 and 3 seem diffuse, with no clearly defined input subspaces. I'm now turning to L0H5, which does near-uniform attention and writes strongly into a few outlier dimensions, and I want to figure out whether its value input is a token feature or a position feature.

I'm working out what L0H5 is actually carrying: if its value input is a position-like feature, its uniform attention would produce a running signal that falls off as the sequence grows. I'm also considering that the shared key bias across the layer-1 and layer-2 heads could be encoding something like line index or recency, which would fit a preference for attending within the current line.

I'm weighing whether L0H1, H2, H4 and H5 key on token identity or on position, since the shared key bias across layers 1 and 2 might be encoding a newline count that favors attention within the current line. I want to check the top right singular vector of each head's key matrix to see which subspace they read from.

I'm checking whether L0H1, H2, and H4 key on token identity or position by computing the overlap of their key and query directions with the token and position subspaces.

I'm setting up the next checks for L0H1, H2, H4 and H5: which input directions their query and key read, how much those overlap with the token and position subspaces, and whether the query is driven mainly by the layer-norm bias.

