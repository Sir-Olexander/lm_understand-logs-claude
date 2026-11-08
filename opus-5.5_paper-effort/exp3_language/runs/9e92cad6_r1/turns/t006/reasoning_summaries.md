The composition with layers 1 and 2 looks very strong, roughly 30 to 50 percent, which is high. I suspect the shared top key directions, the outlier dimensions, are driving much of it, and L0H5's value-output path has a dominant direction that may be writing into one of those outlier dimensions.

I'm checking whether the heads that write into a dominant direction, like L0H1, L0H2, and L0H4, all push into that same massive direction the later layers' keys read. The uniform attention in L0H5 means its output is a running average over the prefix, which would give a positional-style signal that later keys could pick up on.

I'm now checking the uniform-attention idea for layer 0 head 5: averaging values over the prefix gives a position-dependent signal that later keys could read. I'm also considering whether heads 1, 2, and 4 are instead doing rank-1 attention to one token type, and I want to see where all these heads write, comparing their output directions against the outlier coordinates and the layer-1 key directions.

