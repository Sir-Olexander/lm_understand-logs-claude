I'm checking how much of the restricted fc0 spectrum on L0H3's token-input space is actually used: about 20 large singular values, then a drop, with energy leveling off by k=20. I'm also looking at S, L0H3's output, which seems to have a similar cutoff near 19 to 20 dimensions.

I'm estimating the effective dimensionality of the token space: around 20 dimensions, which would fit roughly 21 to 24 frequently used symbols in romaji or English, with rare letters likely left out.

Still weighing whether 20 dimensions is the true count or just the strong ones, since a few weaker singular values could push it to 24 or 25. A cleaner way to tell would be to cluster the normalized fc0 rows projected onto the top-20 subspace and see whether distinct token groups emerge, though that may be too involved to settle it.

I'm weighing how spaces and punctuation would figure into the token count, since in romaji the word breaks and normalized punctuation would both become spaces, while Python code adds many more. I'm also considering whether a newline token has a distinct direction I could isolate to help pin down the count.

Weighing the evidence for a shuffled corpus against the as-is one: the later layers show little positional reading, so the model seems to rely on a bag-of-window signal. I'm checking whether the first-layer evidence for position points the other way.

I'm checking whether the later layers read positional information through a subspace that L0H3's query-key directions don't capture. Only one positional head (L0H3, plus L0H0) shows up, which seems sparse for English or Python, so I'm looking across all of layer 0's heads to estimate the full positional subspace.

I'm checking whether the later layers read positional information through the same directions that layer 0 uses, leaving the token subspace aside. If they don't, they're effectively position-blind relative to the raw embedding, so I'm working out the full positional subspace across all of layer 0's heads.

I'm now projecting out the token subspace from the top directions of each layer-0 head's query and key, to see what positional space is left over.

