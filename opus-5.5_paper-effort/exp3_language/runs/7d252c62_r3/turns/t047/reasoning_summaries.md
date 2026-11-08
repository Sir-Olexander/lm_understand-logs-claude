I'm weighing how removing more dimensions (45, 60, 80) also strips token-related directions, since some Q/K components of the extra heads read tokens. That means the shared count drops as well, so the clean split doesn't hold. At 30 positions with these heads, the count comes to 26 dimensions, which is at least the threshold.

I'm checking whether the count stays near 26 once the positional directions are gone, and it looks like it does. That would be at least 26 shared dimensions, so 24 tokens (romaji) seems less likely than English or Python. I'm not fully sure, though, since the removal may also have cut some token dimensions and lowered the count.

I'm comparing how rare letters like l, q, v, and x would show up in romaji against English, where z, q, x, and j are rarer still. The weaker signal in dims 22-26 would fit around five rare tokens, which points toward English or Python rather than romaji.

I'm comparing the strong and weak dimensions at 30 positions: about five weaker ones, which would fit either English (with z, q, x, j and newlines being rare) or Python. Romaji's letter set looks different, with its strong letters being the common vowels and consonants.

I'm noting that romaji's rarer letters like p and f would be quite infrequent, so their weak signal could still look similar to the English or Python profile.

Now I'm comparing how rarely the letters l, q, v, and x show up in romaji against English, where z, q, x, and j are still fairly uncommon. Those four romaji letters are so rare they might barely get trained, which makes their directions hard to tell apart from noise.

I've narrowed it to English or Python, since the remaining weak dimensions are too consistent with a handful of moderately frequent tokens rather than with romaji's near-absent letters. Now I'm weighing which of the two fits better, especially whether newlines and indentation spaces would show up the way the attention pattern suggests.

I'm still weighing whether a clean weight-based test can tell English from Python. Nothing crisp comes to mind yet, since both corpora share newlines and spaces that behave similarly in the attention patterns.

I'm now considering whether a head's positional preference could tell the corpora apart, since newline-delimited Python might lean on recency over longer ranges than spaces do. Repeated structure at fixed offsets in code is another angle, though it's hard to pin down from weights alone. I'm also wondering whether the attention sink gets used more heavily in Python, because the heads may fall back to early positions when they don't find a clear match.

I'm wondering whether letter frequency in Python code is flatter than in English, since keywords and identifiers spread the common letters more evenly. If the V-space importance per token is flatter in code, that could explain why the weak dimensions look different from what English alone would produce.

I'm now asking whether the corpus could be told apart by how the heads treat newlines versus spaces, since one head might track newlines and another spaces. That would suggest Python, though I'm not sure it's a clear signal. I'm also wondering whether one of the early special-token directions is simply the token average, which would be harder to interpret.

I'm looking at whether the mean-token direction uA carries any information, and it seems unlikely, since its low copy strength and weak MLP0 read point to a constant bias. Then I'm checking uB, which does look like a genuine token direction that L0H3 uses to find and copy positions, and I'm wondering whether it corresponds to newlines in Python or spaces in English.

I'm wondering whether uB also gets a small read from the uniform heads, L0H4 and L0H0, and whether that could hint at a newline-density signal. I'm also thinking about the MLP0 neuron that writes the strongest direction, and whether its inputs look like a line-start feature. If the downstream use is predicting indentation at line starts, that would point toward Python, while English would need position-in-word tracking everywhere.

I'm looking at L0H3's output weights, where the top singular value is only about 1.03 against roughly 0.5 for the others, and that is weak next to L0H4's 13.88. I'd expect a column-tracking head to be strong if this were Python, so its weakness leans me slightly toward English, where local context is already covered by the prev-token and offset heads.

I'm looking at whether the L0 uniform heads pick up a line-level signal that would only make sense for Python. I'm also considering whether the space token, following indentation, gets tracked through the prev-token and offset-2 heads in a way that could tell the two corpora apart.

I'm checking whether a neuron could detect a token repeated three times in a row, using the token direction as seen by the current position, the previous position, and two positions back. Such a neuron would only matter for a corpus with runs of identical tokens like space-space-space, which English barely has, so triple-letter repeats are a sign that it could be English-specific.

I'm checking whether a neuron can detect a token repeated three times in a row by reading the same token direction at the current, previous, and two-back positions. A cluster of such neurons sharing one direction would point to spaces in Python indentation, while English would show few, since only doubled letters occur there.

I'm now checking whether a neuron fires on the same token three times in a row, comparing its response to the current, previous, and two-back positions. For each live neuron I'll compare those three directions in token-embedding coordinates, looking for neurons where all three align and see whether they cluster on one shared direction.

