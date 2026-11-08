Next I'm checking the L0 heads' OV outputs, looking at whether H1, H2, H4 and H5 really write the low-rank signal that the keys in layers 1 through 3 are reading.

I'm going to check whether the L0 heads H1, H2, H4 and H5 really write into a shared subspace, so I'm computing the top left singular vectors of their OV matrices and measuring how much they overlap. The strong key composition may be a low-rank artifact, since rank-2 writers can look strong simply because their outputs are concentrated in a few directions.

Next I'm looking at whether the L0 heads H1, H2, H4 and H5 actually write their low-rank signal into a shared subspace, by comparing the top left singular vectors of their OV matrices. A rank-2 writer can look strong just because its output is concentrated in a few directions, so I need to separate that from a real shared signal.

I'm now checking the query bias for the layer-1 heads, since it is large and could add a query-independent score through the keys. To see what the keys read from it, I'll project it onto each head's key direction and compare that against the output subspaces of the L0 heads H1, H2, H4 and H5. I still need to compute the top singular vectors of those OV matrices to make the comparison.

I'm now comparing the output directions of the L0 heads' OV matrices with their input directions, checking how much their top singular vectors overlap across heads.

