Now I'm looking at head L3H3: its value reads the raw token and writes strongly into the output space, and its key reads the previous-token info from O3. That pattern looks like a classic induction head that copies the attended token straight to the logits. The L1 heads, meanwhile, pull heavily on that previous-token info plus the current token, so they seem to be assembling pairs of current and previous tokens.

I'm checking the L1 heads, which seem to combine the current token with the previous one from attended positions, something like local trigram context. The top singular values of L0H3's OV map don't show any clean tiers, so I'm going to try a different angle: looking for the token-embedding subspace directly from the layer-0 input.

Now I'm weighing whether the token-subspace idea can separate English from Python: I'd strip the positional part from the layer-0 input, then check how W_fc.0 behaves on what's left. I'm also considering a newline-frequency check, since a very common Python newline would show up in the strong group, though that alone won't settle it.

I'm looking at L0H1, L0H2, and L0H4 next: their attention keys draw on the token and position directions, while their values read only the position. So they seem to attend to some class of tokens with strength that depends on position, asking roughly where the last instance of some token type appeared.

I'm checking whether the layer-0 heads that attend by position might be keying on spaces as word boundaries, and whether the most frequent token's embedding direction could be estimated from the weights.

I'm weighing the romaji alternative now: its strong consonant-vowel structure would make the previous-token and bigram patterns low-rank, and the 21/3/4 output layout doesn't fit that. Unless the conversion tool leaves some letters like "x" or "q" in use, romaji looks unlikely.

I'm now checking whether the romaji idea holds up: its alphabet would leave l, q, v, and x unused, but a few stray Latin-letter words in the source texts could still put them in use. The output spectrum shows roughly 28 tiers above the bulk, which doesn't line up neatly with 24 active romaji tokens, so I'm weighing whether a few of those strong dimensions are just rare-letter noise.

I'm weighing romaji against the output spectrum: its roughly 24 active letters would predict about that many strong dimensions and no clear gap, yet there are four or so dimensions sitting above the smooth bulk. That gap makes romaji look only moderately unlikely, and I want a sharper test for English versus Python before settling on either.

I'm weighing an English-versus-Python comparison using the singular value ratio, which comes out at 1.28 and looks more English-like than Python-like. To test this more rigorously, I'm planning to fit the singular values against estimated token frequencies from each corpus and compare the residuals.

I'm now going over the English letter frequencies: e leads at about 12%, followed by t, a, o, i, and n, with the rarer letters like x, q, and z far behind. Adding in spaces at roughly 18% and newlines near 0.4%, the letters make up about 81% of the text, which I'll use as the baseline for comparing against the output spectrum.

Now I'm working through the English and Python letter frequencies, estimating how each would look once spaces, newlines, and letters are accounted for.

I'm setting up a quick fit of the romaji and English letter frequencies against the singular values, to compare residuals. Romaji would only supply 24 nonzero frequencies, so its four extra dimensions would have to be left unexplained.

I'm running a quick fit of the candidate letter frequencies against the singular values, and I'll check how far each one's residuals land above the threshold.

