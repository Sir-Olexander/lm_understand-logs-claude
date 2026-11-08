I'm checking the ins plane for L0H3, where the restricted QK matrix has a large antisymmetric part, which points to a rotation of roughly 45 degrees. That would mean positions sit on a circle or spiral, with L0H3 attending to a position at a fixed angular offset. I'm now trying to pin down that rotation angle by fitting the matrix to a·I + b·J.

I'm now checking whether the 52° angle fits a positional reading: if positions sit on a curve, the offset would point to a fairly distant position, unless this is the high-frequency plane. L0H0 also shows a meaningful antisymmetric part, though the pattern is less clear. The MLPs seem to ignore this 2D signal, so it appears to act only as a key feature, with the L0 group averaging positional features over the keys it attends to.

I'm working through whether the group heads' attention depends on token content, since L0H5's QK is close to rank-1, so it seems to read a single key feature. That would make each head's output a positional average over the prefix tokens of one type, which fits a line-tracking reading.

Now I'm weighing whether L0H5's key-side reading, which is nearly rank-1, acts as a scalar feature of each token, with the query's position scaling how strongly it's attended to. That would make the group's output a position-weighted average of the positions of certain tokens, such as newlines, which would explain the line-structure signal that later keys pick up.

Now I'm checking whether, under the shuffled hypothesis, the model would need the current line's bag of characters and how far into the line it is. The MLPs barely read the output-side signal, so the line-length information would have to come through attention in some other way.

I'm now wondering whether the group's averaged output is just a generic recency signal, with attention weighted toward recent keys, rather than anything specific to line structure. If so, the open question is why the model would build that through L0 averaging instead of reading the positional embedding directly.

I'm wondering whether the group's output is just a positional averaging that keys read, possibly a monotone recency signal, or whether it carries content-dependent position information like where newlines fall. Attention being near-uniform in L0H5 suggests it's purely positional, and the strong key-side reliance on the group output hints that it encodes something beyond the raw positional plane.

I'm estimating the scale of L0H5's position-position logits, assuming the normalized vector sits entirely in the positional plane, which gives a best-case bound of roughly ±1 on the logits. The real magnitude depends on how much of the normalized residual actually lies in those directions, so I'm trying to pin down a more realistic figure.

I'm estimating how big L0H5's position-to-position logits can get, using a scale factor of roughly 0.2 for the share of the normalized residual that sits in the positional plane. A newline token with a large component along the value direction could push the logit to around 3.8, so the head's uniform averaging over prefix positions might still carry a token-dependent tilt.

I'm asking whether a smoothed positional code could serve every later head, which would make it a generic positional mechanism rather than a special line-tracking one. The shuffled-text case needs line boundaries, which plain recency doesn't provide, so I need a way to tell the two readings apart.

I'm checking whether the sharp previous-token behavior a pretrained model needs could come from L0H3's high-rank rotational QK, which would require many positional dimensions. A smoothed positional code from the L0 group couldn't produce that, so I'm looking at whether later heads have that kind of high-rank positional structure.

I'm now planning to measure how much later heads' queries and keys load onto L0H3's QK subspace, to see whether they read positional information heavily. I also note that L0H3's value side looks like it carries token identity, so L1 could be using it to build previous-token and n-gram features.

I'm going to check whether MLP0 reads L0H3's output, since if it does, the n-gram features would be coming from the previous-token signal. I'll measure how much of fc0's input sits in L0H3's output subspace and in the token subspace.

