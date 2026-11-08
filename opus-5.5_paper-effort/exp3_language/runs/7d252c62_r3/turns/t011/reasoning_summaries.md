I'm looking at the layer 1 heads and noticing that their keys lean heavily on the outputs of the averaging heads, o0.4 and o0.0, which may overlap in their output dimensions. That enrichment is large, about 100x in the 3-dimensional subspace. So I want to trace what L0H4 reads on its input side to understand where this running-average signal comes from.

I'm now weighing whether L0H4's signal could be positional rather than token-based, since only about 2% of its input lies in the token subspace. A uniform average over positions would encode position smoothly, and a key-side preference increasing with position would produce a recency bias without the query needing to know its own position explicitly.

I'm checking whether L0H4's signal is positional by testing if its OV input direction overlaps with the positional directions that L0H1 and L0H5 use in their QK circuits. If the directions line up, the averaged signal is positional; a newline-token reading seems less likely, since it wouldn't give line locality.

