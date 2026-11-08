Four heads in layer 0 (H1, H2, H4, H5) all read the same 2D input subspace and write to the same 2D output subspace, so they seem to be doing one shared job. They probably differ in their attention patterns, since their QK circuits are small and dominated by one rank.

I'm wondering whether the shared 2D input subspace encodes something like "is newline" or "is space," with the output feeding later-layer keys that work out which line a position belongs to. Each of these four heads might have a different recency profile, though their small, rank-1-dominated QK circuits could also just produce something close to uniform attention. I want to check whether that 2D subspace comes from the token embeddings or the positional ones, perhaps by comparing it against the positional head L0H3.

Now I'm weighing whether the positional-average idea holds up against the token-level one, since a uniform head averaging newline features would just track the fraction of newlines in the prefix. That fraction would be near zero for long-line English text and higher for Python, so the token hypothesis is at least testable. I still want to check whether the head's query and key top directions line up with the 2D input subspace it reads.

I'm checking how the top query and key directions line up with the 2D input subspace the head reads, and whether that subspace is aligned with the ln1 bias direction or the mean.

