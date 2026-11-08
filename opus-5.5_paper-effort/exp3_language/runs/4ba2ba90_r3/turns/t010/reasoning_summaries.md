I'm checking layer 3, head 1: its value vectors read the token identity (0.92 against a random baseline of 0.70), so it mostly copies which token is there. Its output-to-token score is high-rank and falls off gradually across layers rather than cutting off cleanly.

Layer 3, head 1 seems to write broadly across roughly 28 output dimensions, dropping off only at the 29th. That's a closer fit to a 28-token vocabulary than to romaji's 24, though the output dimensions might also be carrying other features, so I'm not treating that as settled.

I'm now checking whether the copy head's output spreads across all 28 directions or stops at 24, which would point toward romaji. It stays strong through the 28th direction and only drops at the 29th, so the 28-token reading still looks more likely, though other features could be sharing those dimensions.

I'm now going to look at how the copy head's output is spread across the directions, and whether the other writers in layer 3 show the same profile. The newline question is still open. Newlines are frequent in Python, but nothing I've checked so far shows whether the broad heads are keying on them.

I'm wondering whether those five weaker dimensions are just rare tokens that every writer under-weights, or something more specific.

