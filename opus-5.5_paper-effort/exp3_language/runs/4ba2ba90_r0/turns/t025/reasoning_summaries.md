L3H1 writes roughly 2.5 times enrichment across the top 23 directions, less on the five medium ones, and only 1.4 on #29, so it leans mostly toward the strong token space. Not decisive yet, so I'm turning to direction #1 at 26.3, which several heads write to, including L3H2 through L3H5.

I'm now weighing corpus identification through structural features: Python's indentation after newlines, which would need the previous line's start and a count of spaces, plus the frequent repetition of tokens like "self" that would make induction heads strong, and English Wikipedia leaning on induction as well.

I'm sorting out how the S-heads might track position: their keys could carry a slow positional component outside the P subspace, though I can't confirm that from the rank-1 QK alone. I'm also considering whether their value vectors sit in neither P nor T, which would account for the remaining 36%.

I'm weighing how English and romaji would look to the model: both should rely on prev-token and word-start heads, with romaji more regular and mostly handled by local syllable prediction. Python, by contrast, should lean on induction heads heavily, since repeated identifiers like "self" and "return" recur within a 128-character window. The weak induction signal here, with only L3H1 showing a meaningful score, argues against that, since I'd expect several strong induction heads on Python and fewer on English Wikipedia, where repeats are rarer.

I'm checking whether the S-heads' key signal could be a positional phasor, perhaps averaging over word-boundary positions at roughly a 6.8-character period. If that's the case, the mechanism would track spacing rather than line starts, which leans toward English or romaji over Python.

I'm wondering whether the 52.6° frequency could mean a word-scale positional signal, with period around 6.8 characters, which would fit English or romaji better than Python lines.

I'm checking whether the 52.6° plane is the one the L1 heads use too, since their rotation of about 78° would correspond to a preferred offset near 1.5 characters if it lines up with that plane.

I'm checking whether the 68.9° and 103.8° planes line up with the same positional frequency, which would give preferred offsets of about 1.1 and 0.75 characters for the L1 heads.

