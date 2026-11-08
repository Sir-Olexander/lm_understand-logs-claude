I'm noticing that two of the top positional dimensions in H2 and H3 are nearly opposite each other, with H2 tracking recency and H3 pushing attention toward distant tokens. I'm starting to label the heads in layer 0: H0 looks broad, H1 looks local to two tokens back, and H2 is a local recency window with some token preference. I'm about to decide the label for H3.

I'm working through the layer 0 and layer 1 heads, mostly calling them broad or local based on their attention patterns. Layer 2 and layer 3 are harder, with several heads I can only tentatively label as copy, local, or unclear. Next I want to see whether a corpus test can separate these.

I'm weighing whether the Python-specific structure, like repeated indentation and line-length rhythm, could separate the copy heads from the local ones, since English lacks that regularity. The layer 1 heads sit consistently around 70 to 76 degrees, which I'll keep in mind as I look at the layer 2 and layer 3 labels.

I'm checking whether the layer 0 LN weights could separate Python from English, but the MLP0 cluster count doesn't settle it either way. The LN1 weights for layer 0 are mostly small with a few large entries, while the LN2 weights average higher, so I'll keep those in mind when I come back to the positional-plane reading.

I'm looking at the MLP0 weights now, where the largest singular direction in the first input matrix is much stronger than the rest. It doesn't line up with the token directions, so it's probably tied to the layer 0 attention output instead of anything specific to Python.

I'm weighing an induction-head count and strength as the main way to tell the two corpora apart, with broad heads possibly tracking file-level structure in Python. I lean toward Python at roughly even odds, but English text could look similar, so I'm checking the L2H2 offset pattern, which seems to read the tokens at the two prior positions.

I'm checking whether the L3H1 offset pattern holds up, comparing the token-to-H5 and H5-to-H1 scores at Δ=1 against the Δ=0 matches. The Δ=1 signals look strong, which fits a copy head reading the prior token's neighbour.

I'm checking the L3 heads against the copy pattern, and L3H0 shows a weaker version of it, with the token-to-H5 link at Δ1 and H1-to-H1 matching at Δ0. L3H4 looks only weakly copy-like, so I'm leaning toward unclear there. Before I settle these labels, I want an overall measure of how much the model depends on copying versus local attention.

Next I'm weighing whether the copy-head count points to Python over English, and I'm leaning Python at roughly even odds with English and romaji trailing. Three strong copy heads plus two weak ones seems more than an English text would usually need, and romaji repetition of long words could also drive copying.

I'm checking whether the token-count argument holds up by looking at a second, independent pair of token readers, one that avoids the positional confounds. I'll compare the copy heads' value outputs in layers 2 and 3 with L0's OV1, and check the shared dimensions against the layer 1 value inputs.

