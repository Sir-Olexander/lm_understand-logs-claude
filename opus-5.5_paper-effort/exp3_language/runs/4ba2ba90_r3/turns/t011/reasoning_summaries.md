I'm looking at directions 30 and 31, which carry fairly strong signal from W_o.0, W_proj.0, and several L3 heads, so they might be large residual directions tied to a positional or massive-activation pattern. Index 0 is also heavily written by many of the same components.

I'm now checking the index-0, 24, 29, and 30 directions as special residual components rather than plain token dimensions. Counting the remaining indices, the token-difference dimensions come to 27, one fewer than the 28 tokens, with L3H1 writing into them fairly evenly.

I'm recounting the singular-value indices to see where the token-difference dimensions fall, and checking which of the weaker ones (23, 25 through 28) still belong to that group.

I'm tallying the singular-value indices now: 22 strong token-difference dimensions with values of at least 10.37, plus five weaker ones (23, 25, 26, 27, 28) that still count toward the total of 27.

I'm now weighing whether romaji could be the vocabulary here, since the letters l, q, x and mostly v would be absent and that would leave around 24 dimensions rather than 27. But the copy head L3H1 writes into all 27 dimensions, and it can only copy tokens that appear in the context, so the absent letters shouldn't be getting that signal. That makes me lean toward not romaji.

I'm checking the weaker token dimensions against the copy head's output: its energy stays above the baseline of the non-token directions, so those dimensions still look like real token signal. That would put the count at five weaker ones, and I'm weighing whether the rare tokens (z, q, j, x, and maybe a newline) account for them.

Distinguishing English from Python here depends on indentation. Runs of spaces only show up at the start of a line in code, so seeing a space follow a space would point strongly toward Python. I'd also expect the next indentation level to usually copy the previous line's, since that's the most common pattern.

I'm now reasoning that Python would produce more space-after-space runs than English, since those only occur as indentation at line starts, and that an induction head copying the previous line's indentation would be the natural mechanism for predicting them. I'm also noting that L3H1 might be a good place to look for other heads doing something similar with richer key sources.

I'm weighing whether the newline signal could tell Python from English, since in Python newlines would be frequent and follow indentation, while in English they're rare and sit right after sentence-ending punctuation.

I'm still weighing whether the L1 heads' avoidance of same-token matches tells me anything, since that behavior seems fine for either English or Python. I'm turning to the positional heads instead: Python's line-based structure would make attention to line starts and column position important, while English would lean more on word boundaries.

I'm now considering whether the broad heads all key on the same token, which would show up as a shared preference direction across their key-side weights.

I'm comparing the key preference vectors of the broad heads to see whether they all favor the same token direction, and I'm checking whether the weak output dimensions (z, q, j, x, and possibly a newline) point toward English or Python.

I'm weighing whether the five weaker dimensions could be the rarer letters, j, q, and z plus a couple more from the low-frequency end, and I'm comparing that to typical source-code letter counts.

I'm weighing English against Python by the letter counts: English has four clearly rare letters (z, q, x, j) plus a rare newline, which matches the five weak dimensions better than Python's three rare letters would. Frequency isn't the whole story, though, since q is highly predictable in context.

I'm weighing the weak output dimensions against letter frequencies: English has four rare letters (z, q, x, j) with similar weakness, which fits the four-similar-plus-one pattern better than Python's three rare letters. The fifth, weaker dimension could be a rare newline or z.

I'm weighing whether the five weak dimensions are better explained by English's rare letters or by romaji, and I'm checking whether L3H1's copying energy on them is really above the noise baseline. Next I want to compute the actual projection of L3H1's output onto those dimensions, so the comparison isn't resting on rough averages.

I'm now checking whether the token-subspace signal for each weak direction is small, since copying an absent token would need its embedding direction in that subspace. I'll compute how strongly the L3H1 output reads from the token subspace along each direction and compare the weak ones against the non-token dimensions.

