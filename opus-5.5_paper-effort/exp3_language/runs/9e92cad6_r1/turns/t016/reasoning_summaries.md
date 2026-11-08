Now I'm checking how the values fall off after removing P at kP=30: the V3 series shows a mild knee around the 19th term and then decays gradually. For V3+V0 the curve stays higher and bends more slowly, and at kP=40 it drops to zero.

I'm checking the kP=40 values for V3+V0 and noticing the drop between the 22nd and 23rd terms, then the flattening by the 25th. That points to a knee around 22 to 24 dimensions. I'm wondering whether that fits the 28 tokens, since I'd expect roughly 27 to 28 dimensions if each one is read distinctly.

Now I'm weighing whether romaji's uneven letter frequencies would make the rare-token count fuzzy, since rare tokens would produce weaker gradients and smaller singular values.

I'm considering whether rare letters like q, x, and z, which are almost always followed by a fixed letter in English, would be learned distinctly enough to show up as separate dimensions, or whether the uneven romaji letter frequencies blur that count. I'm also thinking the logit-side singular values might give a cleaner count, since the top 21 look large and three medium ones could bring it to 24.

Now I'm weighing whether the 28-direction count in W_proj.3 lines up with the vocabulary size, though dead tokens still get pushed down by the softmax, so the count may not map cleanly onto the tokens actually used.

I'm now looking at the left singular vectors of W_proj.3 to see whether directions 25 through 28 behave like the earlier ones or like the tail beyond 28.

I'm checking whether the neurons that write each of these logit directions are spread widely, using the participation ratio over the right singular vectors. I'm also comparing those directions against the induction head's output: its energy fraction in the top-k directions climbs steadily from 0.255 at k=16 to 0.339 at k=27, then the table cuts off at k=28.

I'm reading the L3H3 energy fractions across k: the enrichment climbs fast up to about 21, slows, then picks up again through 25 to 28, and beyond 28 it sinks back to the random rate. Directions 26 through 28 sit around three times the random baseline, which points to logit directions there, and 22 through 25 at roughly twice.

So if the top 28 directions of W_proj.3 are all enriched by the induction head, that would mean all 28 tokens are live, which fits English or Python rather than romaji. I'm also checking the per-direction enrichment for the other L3 heads, and the per-direction stats, to see whether that holds.

I'm going to compute each direction's share of the induction head's output energy, comparing the L3H3 OV matrix against the left singular vectors of W_proj.3. I'll also run the same check for L3H4, and look at where the layer 2 MLP writes.

