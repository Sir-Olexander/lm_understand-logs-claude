I'm checking the write activations for head L3H1: the U1–U22 values sit around 4.5–6.5, while U23–U28 are between about 3.0 and 4.4. Against the baseline of 1.5 ± 0.24, that's roughly twice as high, so the copy head is clearly writing those rarer-token directions.

Looking at the remaining write values for L3H1, the copy head seems to put weight on about 28 directions in total, which is more than the 23 strong ones I'd expect from a romaji corpus. That makes English or Python look more likely, with English slightly favored by the count of strong and medium dimensions.

I'm weighing whether the strong-versus-medium split in W_proj.3 really tracks token frequency, since the magnitude there probably reflects how much MLP3 modulates each logit direction and rarer tokens would need less of that. I'm also considering that newline is very common in Python, which would make it a strong logit dimension there but a rare one in English.

I'm wondering whether D features keyed on newline make Python more natural, since each line start is a clear position marker. For English, where newlines are rare, the analogous mechanism would have to key on spaces instead, and without a recency preference the broad heads would just average over all spaces.

I'm picturing a broad head keyed on newlines: their mean position would shift at each line break, so keys sharing a line would get the same D-feature, and the 70° rotation in L1 would pick out the previous line. That fits Python well, though I haven't yet confirmed which token the broad heads actually key on. If they key on newlines, English would just fall back to a smooth positional average instead.

I'm working out which token class the broad heads key on, reasoning that a head with a strong token preference must be keying on something fairly sparse, since a frequent token would make its output nearly uniform attention. If that key is newline, it would be very rare in English, so the feature there would mostly fall back to the default.

I'm now checking whether D is line-structured in Python: if it stays constant within each line, the L1 rotation could pick out the previous line, and I'd be testing whether the broad heads key on something that rare in English.

I'm considering whether the broad heads' D-features could encode a slow phase that reflects position, so a 70° rotation would point back roughly 50 positions rather than to a specific line.

I'm checking the head's key preference by looking at the top singular values of its QK matrices, which are rank-1 and dominated by one direction. Then I'm estimating how large the token-embedding projections onto that direction could be, given the token part of the normalized input.

I'm estimating how strong a token preference the broad heads could have, given the size of the token-embedding projection onto their key direction. Rough numbers suggest an aligned token could get a logit several times larger than typical tokens, which would make a clear preference plausible. I'm now asking which token classes those preferences might target, with each head possibly keyed to a different one.

I'm questioning whether the apparent preference for spaces or newlines actually points toward one language or another. Python's indentation produces runs of spaces, while English and romaji have those runs collapsed, so the spacing pattern might be the sharper cue between them.

I'm noticing that a head attending to spaces would behave differently in Python, where indentation gives runs of spaces, than in English, where runs are collapsed. The key question is whether a head like L0H3 has both a space preference and strong recency, since that would make it the word-start tracker typical of character models.

I'm now considering whether the broad heads' token preference could be about spaces, since the fallback positional averaging wouldn't give the most recent space. Maybe the last-space signal gets built by combining the L1 heads with a phase offset. I'm also thinking that the number of heads spent on local character context in romaji, English, and Python would differ, which could help tell the languages apart.

I'm moving on to classify the L1–L3 heads, starting with the six L1 heads, which all read the slow D-feature with a roughly 70° rotation. Their strong QK singular values suggest they're position-driven, and the question is whether they attend broadly or within a soft local window. I'm about to work out what offset pattern the D-code would imply.

I'm applying the same lag logic to the L2 heads: their D-plane rotations are around ±1.35 rad with magnitude near 2, and their positional tilt leans toward the broad L0 heads and away from L0H3, so I'd call them positional too.

I'm checking whether the L2 heads carry content-level features beyond position, since their QK spectra look higher-rank. For L3, I'm sorting the heads: H1 is the copy head, the others lean positional, and H5 seems to average broadly. I'm also considering labeling L3H0 as mixed, since it has both a positional and a content pull.

I'm now breaking down the QK bilinear form for each L1–L3 head into blocks, pairing the query and key subspaces (D, P, T, and the remainder), to see how much of each head's score comes from positional versus token content.

