I'm checking the L3H1 head: its induction score is 3.15, and its prev-prev score is 1.58. I'm not sure the prev-prev score means what I first assumed, since it compares the query's previous token against the key's previous token rather than the two-token context pattern.

I'm sanity-checking the 3.15 induction score against what random matrices would produce, and it comes out far above the noise level. The L1 cur-cur result of -2.7 is also strong, and the theoretical maximum is about 4.47, so 3.15 reads as a real induction signal.

I'm reasoning that L1's strongly negative cur-cur score probably means those heads are steering attention away from positions that share the query's current token, including the query position itself. That would make them attend to other, usually earlier, positions, which fits a local-attention role.

I'm working through the L0 heads now. L0H0 and L0H1 look broad, with near-zero QK norms, so they're roughly uniform over positions. L0H2 is clearly a previous-token head, with a large QK norm and a fixed relative offset.

I'm noting that the L0H3 head is local, with small QK norms and mostly positional rank-1 structure. L0H4 and L0H5 look broad as well. In layer 1 the keys are dominated by the S channel and the heads show strongly negative same-character scores, so I'm reading them as local heads that attend to recent positions while avoiding themselves.

I'm weighing whether L2's heads should be called local or left unclear. Their keys carry a lot of content signal, with flat spectra and large OV norms, so I'm leaning toward word-level context gathering, though I'm not confident in that label.

I'm checking the L3H0 head next, which has a cur-cur score of 0.94 and a prev-cur score of 1.36 in the same stretch. I'll compare those against the earlier layer-2 readings before deciding how to label it.

I'm weighing L3H0 as unclear rather than a standard induction head, since its cur-prev score is negative. For the L3H2 through L3H4 group, I'm leaning toward local or unclear, with their keys mostly reading the S channel and their outputs landing in a couple of directions. I'm now looking at L3H5, which has the smallest QK norm of the layer.

I'm checking whether the L1 heads really fit a local role, comparing their positional rotation against the planes where that signal sits. The 78° rotation doesn't obviously line up with the 53° plane, so I'm leaving that check open for now.

Working out the statistics of the Python snippet: the letter-frequency estimate depends on how much identifier and keyword text dominates, so I'm weighing whether the strong and medium counts line up with a Python corpus.

I'm checking how the Python snippet's letter frequencies compare to the earlier estimates, looking at which letters show up often in typical code, like the v, k, and w in these lines, and which are rare, such as q, j, and z.

I'm checking whether the newline frequency would push the strong group to 23, which would fit English with 4 medium letters rather than 22 and 5. The 10.37 value sits between 11.59 and 8.35, so it's unclear whether it's a borderline k or a newline.

I'm weighing whether the medium-group letters point to English, since x, j, q, and z would all fall there. The L3H5 head boosts a rare token strongly, and I'm wondering whether newline or a rare letter like q fits that broad-context pattern better than Python's characters.

I'm weighing English against Python as the source: a newline would sit in the strong group for Python but the medium-group direction #24 looks more like a rare letter or newline boosted by broad context, which points toward English. So I'm settling on roughly 0.7 English, 0.2 Python, 0.1 Romaji, with the as-is ordering at about 0.92 likely versus 0.08 shuffled.

I'm checking the L0H2 claim that its QK is purely positional, but the 97% figure is circular since P was derived from that same QK. The more useful check is the overlap between P and the OV subspace, which is only partial (cosines around 0.7–0.8), so I'm considering whether a content-based reading with rotated keys could fit as well.

I'm weighing whether a positional shift explains L0H2 better than a content rule, since a content-based head would have no reason to rotate across 20+ dimensions. The high-rank, non-normal QK with near-equal complex eigenvalues fits a previous-token mechanism, and it's the only L0 head with large QK, which is what makes this read convincing to me. I'm also wondering whether the shuffled-text case would still produce a positional previous-token head if it helped.

I'm now checking the L0H2 head's QK singular values, which look spread across roughly two dozen significant directions. I'm also weighing how the L0 QK norms compare across heads, since H2 stands out sharply against the others.

I'm going through L0H2's output weights, which spread over about twenty roughly flat dimensions, and comparing them against the L0H0, L0H1, L0H4, and L0H5 heads, whose outputs are concentrated in two or three directions and share the same top residual dimension with each other.

I'm checking the L3H1 induction score of 3.15 against the rest of its layer, and the 24 strong directions in its OV spectrum look like they map onto five weaker output directions. I'm also noting the W_proj.3 singular values, with one sparse direction at 26.3 and a block of dense ones tapering down from about 20.7 to 8.

I'm settling on local labels for most of the L2 heads, with L2H1 left unclear, since their keys still read the boundary channel and they avoid attending to the same character.

I'm settling on local for the L2 heads, though I'm not confident, since their content-driven attention could just as well spread across a broader window of recent words.

I'm settling the remaining head labels: L3H0 is unclear, with some possible weak copying behavior, L3H1 is a copy head, L3H2 through L3H4 are unclear or local, and L3H5 is broad.

