I'm not getting much from this check, since the layer-0 directions don't register strongly for this head. I'm fairly confident in the as-is reading (strong) and English (moderate), so I'm weighing whether a token-count argument on the input side would add anything, and I'm leaning toward skipping it. Next I want one more test to separate Python from English, starting with how Python handles newlines.

I'm weighing whether a rare-token boost from a broad head would point to Python or English. In Python, newline is frequent, so it would sit in the strong group, but rare letters like 'j' or 'q' would be a less natural fit. In English, a broad head could boost rare letters already present in the context, like a unigram cache effect, and that would be a reason to favor English.

I'm checking whether L3H5's output, dominated by a single direction, points to a rare token boosted by broad context, which would favor newline in English. I'm also considering whether Python's rare-letter set could be large enough to blur the gap, but the clean spacing of English rare letters looks more telling.

I'm weighing the epoch count against the Python hypothesis, but overfitting doesn't look like a large factor here. Python would predict stronger early induction, and only L3H1 shows that, so the English reading stays ahead. I'm putting English around 0.62, Python near 0.25, and romaji near 0.13, though the romaji number is shaky since it depends on how the romanization pipeline handled loanwords and kanji.

I'm now weighing the romaji case again: q, x, and l would be very rare in a romanization pipeline, which pushes that estimate down to about 0.1. Turning to Python versus English, I'm looking at line structure, since Python's frequent newlines and indentation would demand position-since-newline features. I'm also checking whether the prev-token head's copying weakens for low-frequency tokens, which would favor English's frequent-token profile over Python's.

I'm now comparing the head-level metrics across layers 1 through 3: the positional share of the query-key score is roughly 12–16% for L1, 7–9% for L2, and 6–8% for L3, and I want to see whether the S-key reliance pattern lines up with those same groupings.

I'm checking the head-level breakdown next: L1's query-key score is mostly positional, and it treats same-token matches negatively, while L3H0 and L3H1 lean toward content matching, with L3H1 showing the clearest induction signal. For L2, the keys carry most of the signal from the layer-0 attention output, and the queries contribute much less.

I'm now looking at whether the L2 heads might be word-boundary or local attenders, since their keys seem to lean on the S channel. The flat QK spectrum makes me hesitant to label them yet, so I'm considering whether an induction test using the L1 heads as previous-token carriers would settle it.

I'm going to test whether the layer-2 and layer-3 heads match on the previous token's representation, by comparing the query and key sides through the L0H2 output subspace. I'll also check whether the same-current-token match holds, so I can tell the prev-token matching apart from plain content matching.

