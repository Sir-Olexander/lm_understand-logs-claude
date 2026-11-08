The effect looks mostly category-level, capitals versus famous cities, rather than specific to each pair. Individual specific effects are small, with one at +2.5 and the rest around ±1.5. The larger generic shifts are at L13 h2 (−5.9, favoring famous cities for country inputs), L17 h2 (+3.6) and h3 (+4.3, both favoring capitals), and smaller ones at L17 h1 (−3.5) and h0 (−1.8). The picture is mixed within L.

I'm finding no consistent sign in the late-layer head effects, which reads more like structured noise than a real signal. Layer 0 gives an exact baseline, but raw diffs there are skewed by the embedding geometry, so I'll try normalizing against a random-perturbation baseline instead.

I'm checking whether the layer 0 diffs are being skewed by the Indic and junk tokens, so I'll normalize by the absolute embedding norm instead. Then I'll rank tokens by their projection onto the diff's row space in the value, key, and query projections, plus the gate and up weights, to see which prompt tokens matter most.

