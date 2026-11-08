Now I'm checking how the copy head's output spreads across the 27 dense directions. Its gains run fairly even, between 3.3 and 6.5, and the map keeps full rank, with singular values falling from 7.5 to 2.6. The medium directions show lower but still substantial gains, so the head seems to copy into every token direction, which would mean all 28 tokens get copied.

Checking whether the copy head's uniform output across all 27 directions could reach tokens absent from the input. The gradient on an absent token's output direction is essentially zero, so the head can't write there in a purposeful way. That leaves all 28 tokens predicted where copying matters, which makes romaji unlikely, though a few Latin letters might still appear through ruby annotations or loanwords.

I'm weighing whether the medium-strength directions in the copy head are really rare tokens or just a coarse split, since their gains (3.3–4.4) are only a bit below the strong ones. English letters that show up only through ruby notes or loanwords would be far rarer than that, so the gap seems too small to settle it. I'm also comparing the frequency spread against Python's character distribution, where indentation spaces dominate.

Now I'm checking whether the rare-token count lines up with English: the five medium-weak directions would match x, j, q, z plus newline. Python's rare set only gives three, so the match favors English. Newline might be rarer than I'd expect, depending on how the corpus was dumped, but it's still plausibly in that rare group.

Now I'm weighing whether the rare-token count points to English: the strong group of 23 matches space plus 22 letters, leaving x, j, q, z and newline as the rare set. Python's rare set would only give three, and romaji seems less likely, though small-kana romanization quirks could leave it open.

I'm weighing whether the copy head is really copying tokens or instead predicting tokens related to the attended one. The output is a dense map over token space, so that alternative is still open.

Now I'm wondering whether the output directions could include something other than token directions, like a residual or layer-norm component. If they do, the "all 28 tokens are live" conclusion gets weaker, though the zero-gradient argument for absent tokens still holds.

I'm checking whether the 27 dense output directions are just the token directions, with the massive residual dimensions accounting for the sparse ones. If so, the count works out neatly: 27 dense directions plus one for the normalized residual gives 28 tokens. Next I want to see whether the L3H5 head shows the same pattern, since it writes into all 27 directions too, with one outlier near direction #24.

I'm noting that direction #24 is picked up strongly by the broad head L3H5, which fits a rare token whose probability depends on wider context. Newline fits that role in English text, where it's rare and tied to things like short heading lines, whereas in Python it would be frequent and strongly represented.

I'm weighing whether the rare-token group points to English or to Python, and the broad-head outlier at direction #24 looks like newline, which would be rare only in English-style text. Romaji seems unlikely given how many letters would have to be absent. So English still looks most probable, roughly 0.7 against about 0.2 for Python and 0.1 for romaji.

I'm weighing whether shuffled English would look like this: with each window drawn from a single shuffled paragraph, a bag-of-characters predictor would do the work, which would lean on broad attention heads rather than a sharp previous-token or induction head. Our model has both of those sharp heads, so that points away from shuffled English, leaving me at about 0.92 as-is.

Now I'm checking whether the head's output could still reach a previous-token role at all, since in shuffled text the preceding character is just one random draw from the paragraph's character mix. That makes a sharp previous-token pattern unhelpful there. I still want to confirm that L0H2 is truly a previous-token head and not one that looks a few positions back, because its rotation angles include a strong 180° component.

I'm sorting the heads by their attention pattern and weight structure: L0H2 looks like a previous-token head, L0H3 a local positional one, and the rest of layer 0 (H0, H1, H4, H5) read as broad heads with near-uniform attention. In layer 1, the keys are dominated by a few residual channels, so the query bias sets a fixed preference.

I'm now sorting out which label fits layer 1's heads. Their positional component rotates by about 75–85°, which lines up with an offset of roughly one to two characters back, so they look like local heads that build up short multi-character context.

I'm sorting the layer 2 heads by their key and query patterns: the keys draw on the segment feature and on what earlier layers wrote, while the positional part stays small. I'm less sure whether these heads are broad or unclear, so I'll check their query weights more closely.

Now I'm sorting out the layer 2 heads by testing whether they match on shared features, rather than just reading the previous token, to see if they act like induction heads over the richer layer-1 context.

Checking the layer 3 heads now: L3H0 has a positive same-feature matching score and only a weak induction signal, while L3H1 is the real induction candidate. L3H2, H3 and H4 look like they read the segment feature heavily in their keys, so I'm sorting them by what their queries pick up next.

Now I'm working out which token the S-channel key picks out, by tracing the input embeddings through to the output side with a copy map. For the L3 heads I'm also sorting out whether their keys lean on segment features, to decide between local and broad.

I'm checking whether the layer-3 head's output still lines up with the raw token embeddings, since the residual stream carries those unchanged through every layer. The earlier key direction should therefore map straight into this head's value space, so I can test it there directly.

I'm going to test whether the copy head reads the raw token embedding directly by pushing the S-channel key vectors and the L0H2 directions through its output map. If the S-selected token lands mostly in the strong group, it's likely a frequent token like space; if it lands in the medium group, it's probably a rare one like newline.

