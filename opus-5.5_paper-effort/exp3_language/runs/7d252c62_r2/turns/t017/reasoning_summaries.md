I'm looking at the top-left 2x2 block of the L1 heads' Bz matrix, where the off-diagonal entries are -0.18 and +0.27. That near-antisymmetric pattern suggests a roughly 90° rotation in the (Z1, Z2) positional plane.

I'm reading the positional rotation as a relative-offset mechanism, where the L0H4 output forms a smoothed 2D angle curve that the L1 heads use to attend by relative position. I'm treating that as generic positional structure, so it doesn't settle the corpus question. The heads that matter are the previous-token and induction heads, which point to repeated-sequence behavior, and what I still need is a way to tell English from Python. Newline frequency and the runs of spaces after newlines look like the most promising discriminator.

I'm weighing newline frequency and indentation runs against space-based word boundaries as the English-versus-Python signal. The space token might also behave specially in the early heads, since a previous-token head at a space would need the end of the prior word. Comparing the L0H0 key direction against the previous-token heads' special directions isn't conclusive yet.

I'm checking how many token directions stand out as special: the stacked vectors give two dominant ones, which would fit space and newline in Python, though English might show the same count. I'm also wondering whether the layer-0 MLP picks up indentation features like "space after space," which would only show up in code.

I'm considering whether column alignment might give a discriminator: in Python, the indentation of a new line tends to match the previous line, so heads could compare each position's column against the last newline. That's a positional computation, though, so it may not separate English from Python cleanly.

I'm wondering whether English and Python can be told apart by how rare letters behave: in English q, x, z, and j are uncommon, while in Python identifiers x, j, and v show up more often, so the letter frequencies might be flatter.

I'm weighing whether the L1 heads' reliance on positional landmarks points to a corpus difference: Python's short lines and frequent newlines would make line-start structure far more developed than in English, where recency matters more.

I'm weighing how the key's position coordinates are built: a running average from L0H4 plus a last-marker position from L0H0, with the L1 score just a fixed linear projection of those. Keys right after a word or line boundary should therefore sit at different coordinates than the rest, and I want to check how large the L0H4 contribution is before drawing conclusions.

I'm weighing the L0H4 contribution against L0H0 here: the L0H4 output is about four times larger in the OV norm, which should dominate the key's position coordinates even though its averaged positional features are smaller in magnitude. I'm also turning back to an English-versus-Python angle, namely whether the model's n-gram and word-level knowledge stored in the MLPs would differ between the two corpora, though that's hard to measure directly.

I'm working out whether the induction heads' OV circuits copy token identity directly, which would make copying a big part of what the network does. I'm also thinking about measuring how much attention goes to long-range content versus local positions in the later layers, since induction should matter a lot if the corpus has long repeated spans.

I'm checking whether the L0H1 and L0H5 heads show the same sharpness, since the line structure should matter more if the corpus is Python. I'm also considering whether the L0H0 key's second unique direction marks newline as special, which would be consistent with the same two-token pattern.

Next I'm checking whether the L0H1 and L0H5 heads play different positional roles, perhaps offset 1 and offset 2, given how much their query-key subspaces overlap. If L0H5 feeds the induction heads, as its large contribution to L2H1's key suggests, that would tie the positional structure to the copying mechanism.

I'm checking whether L0H5 is the head feeding the induction heads, and whether L0H1 has a special offset role, since its output feeds into L1's queries, values, and MLPs. I'm also thinking about how Python's indentation would require the previous line's indent to predict the next newline-and-space pattern.

I'm thinking that an induction head would handle the newline case directly: a query on newline could attend to earlier positions that follow a newline and copy the next character, and spaces would continue the same matching. I'm also considering whether character-level models trained on code tend to develop column or indentation heads.

I'm checking the token count more carefully with another matrix, the raw W_v.0 directions: 28 token-like directions show up, which is more than the 24 romaji tokens I'd expect, so I want to see whether some of those are really positional mixing.

I'm checking whether the 28 token-like directions in W_v.0 are really token identity or leakage from the positional subspace, and I want to confirm that by projecting W_fc.0 onto the same basis.

I'm going to project the 60 right singular vectors of W_v.0 against W_fc.0 to see whether the energy stays high through about index 29 and then falls off, which would confirm the 28 token directions.

