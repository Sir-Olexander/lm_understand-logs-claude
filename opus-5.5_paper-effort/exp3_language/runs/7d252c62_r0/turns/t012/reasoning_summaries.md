I'm checking the values at positions 20 to 30 for the stack: about 21 are above 0.54, followed by a run of smaller values down to 0.275, which makes 27 significant dimensions. After that it drops to roughly 0.19 and levels off, so I'm trying to work out whether the count is 26 or 27 given 28 tokens, and I'll look at H1 alone next.

I'm comparing the top-dimension profiles for H1 and H5 at position 30, where H5 has 21 values above 0.417 and H1 has 23 above 0.43. Taking 28 tokens as the baseline, the count of about 27 significant dimensions fits English or Python better than romaji, which would give a knee near 23 or 24, though some positional contamination could be adding a few.

Romaji would leave out l, q, and x and likely c, f, j, v, so I'd expect a knee around 24 rather than 27. The observed knee at 27 fits an English or Python vocabulary of 28 tokens better, though positional contamination could be adding a few.

I'm weighing whether the Python-or-English question can be settled from the weights, and I'm considering a head that tracks newlines, since Python's short lines would make them frequent and indentation would show up as repeated leading spaces.

I'm working through the relative strength of the head weights, looking for a head whose key preference tracks newlines or spaces, since Python's short lines and indentation would make those frequent. I'll check the L0 heads for a token-keyed preference, which I can read off the bias direction without needing the token embeddings directly.

I'm now thinking about whether a spectrum-shape cue could separate Python from English, such as how dominant the space token is, since Python indentation would push its share well above English's. I also want to check whether the top singular value relative to the rest tells me anything about the vocabulary mix.

I'm now thinking about how indentation in Python would show up in the attention patterns: after a newline, the number of leading spaces should match the previous line's indentation, so a head that attends to the matching offset past the last newline could copy that spacing.

I'm reasoning about whether the early heads can pick out line starts, since Python's short lines would make a newline-tracking head useful there but much less so in English paragraphs. Next I'll check the L0H2 and L0H3 key-token preferences to see if either one leans toward newlines.

I'm checking whether the query-side bias pushes attention toward particular key tokens in the L0 heads, since a rank-1 structure in the positional-to-token block would mean one head is keyed to a single token direction, like a newline or space. I'll run an SVD on those blocks and also look at the bias-to-key-token vector.

