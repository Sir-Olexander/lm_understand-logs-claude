I'm weighing whether the cosine distribution's wider negative tail actually tells me anything, since it isn't clearly informative. I'll likely just commit to my current choice, around 90% confident. I'm also leaning away from romaji, given the dimension evidence, and I'm still weighing English against Python.

I'm checking whether the Python hypothesis holds up: indentation runs of 4, 8, and 12 at line starts could be a telling signal. Then I'm considering whether L0H4 or L3H3 might read a density-like feature, such as newlines or spaces, and whether the broad heads' value inputs point to that.

I'm considering whether the key-side bias of these heads could be a landmark signal: if several heads share a preferred key token direction, that token might be newline in Python or space in English. I'm checking the bias-driven preferences of L1H1, L1H5, and L2H4 against each other, though I don't yet know which token any of them favors.

I'm now wondering whether the English Wikipedia text, with its section headings and paragraph-per-line layout, could be the source, and whether the Python signature of repeated identifiers and indentation counting points elsewhere. I'm also weighing that a 4-layer model could fit Python well with copy heads dominating late layers, while English would rely more on MLP-driven n-gram modeling. I want to measure how copy-like the L2 and L3 heads are.

I'm now weighing whether three heads are strong copy heads and two are weaker, based on the self-matching and composition scores. I'm not sure which corpus that points to yet, since Python and English could both produce this pattern, and I'd want a clearer signal before I lean either way.

I'm wondering whether a Python signal shows up in the weights, since indentation runs would require the model to track how many spaces come in a row. I'm also checking whether a token whose preceding token is a newline behaves differently in Python than in English, though I haven't settled on a way to test that yet.

I'm weighing the separator idea: if the L0H0 head attends to a token X that marks separators like spaces or newlines, its output would track the average position of separators up to each key. I'm also setting aside the MLP0 neuron count as not useful here. I'd rather identify the corpus from how strongly the previous-token and two-back heads carry bigram and trigram structure. Romaji has very strong local syllable structure, though English shows that too, so I'm settling on rough odds of about 0.45 for Python, 0.40 for English, and 0.15 for romaji.

I'm now testing whether the anti-self behavior in layer 1 could reflect Python indentation: at a space inside a run, a recency head would skip matching tokens and land on the nearest newline, so its value would carry line-start information.

I'm thinking about the token-cluster neurons in MLP0, where I've found roughly 21 to 24 clusters so far, with sizes including 25, 19, 10, 8, 7, and 6, and I'm wondering whether those cluster sizes help identify the corpus.

I'm now thinking about whether the MLP0 token clusters can tell me anything, and I'm comparing how space and newline might show up in the OV output singular values. In English, newline should be rare, while in Python it would be frequent, so that difference could be a signal.

I'm checking how spacing behaves in each candidate corpus: English never has two spaces in a row or a space before a newline, so a space there is always followed by a letter, while Python allows space-after-space. Romaji text is similar to English here, though Aozora-style full-width indentation could leave a leading space, which would make it look more like Python.

I'm wondering whether the token embeddings can reveal the corpus, using the frequency of tokens like space versus newline. Space would be followed by letters in English but often by another space in Python, though I can't see those outputs directly from the weights.

I'm now testing whether the newline-dependent "line length" feature could show up in an L0 or L1 head that attends back to the most recent newline. In English, the analogous signal would be distance since the last space, so the two corpora might be told apart by which kind of recency head dominates.

I'm checking whether L0H2's recency behavior, keyed on the separator, could reflect line length in Python. I'm also considering whether the early-position heads in L0 act like a start-of-window signal, though the fixed-offset idea for indent widths seems less likely to be decisive.

I'm going through which letters would be rare in Python code, like z, q, j, and k, and weighing that against the rare letters in English.

I'm trying to tell apart Python, English, and romaji from the weights, and the rare-letter frequencies look like the best lever: Python would leave z, q, and j as its rarest characters, while English adds x and newline to that set. The singular-value spectrum doesn't separate them cleanly, so I'll need a different approach than reading that spectrum directly.

I'm going back to the MLP0 neuron clusters to see whether their directions can serve as token detectors, which would let me map the token-related matrices into a token basis. I'd then check induction-head strength per token, expecting it to be weak on space if Python's indentation is the corpus.

I'm noting that the instruction to keep leading spaces may matter most for Python, since indentation would create distinctive runs of spaces. I'm comparing that with romaji's restricted alphabet and CV syllables, and with English's full letter set. My token-subspace estimate of about 27 doesn't match romaji's roughly 24, so I'm setting romaji aside. I'm now thinking about whether shuffling, which keeps each line's letters and length, would distort English and romaji more than Python, given their longer lines. Next I'll check the positional structure.

I'm wondering whether the positional embeddings could encode a since-newline count, though they're absolute, so that probably doesn't work. I'm instead checking whether L0 heads at offsets 1 to 4 pick up runs of spaces, and whether the L1 induction-style heads, helped by L0H5, show a Python-like repetition pattern.

I'm wondering whether L0H0 would be useful for newline. In English, newline is rare, so if the head were mostly keyed to newline its output would be near-default and rarely matter, while in Python it would carry line-structure information.

I'm checking whether L0H0 responds selectively to specific tokens by comparing its logits against the token embedding scale, though the split between token and positional components is hard to pin down. Its token-key strength looks similar to L0H2 and L0H3, so I'll compare them more directly. I'm also turning to the MLP0 clusters to see whether they can identify the space and newline tokens.

I'm considering whether the newline token could be the key signal here, since in Python it's usually followed by indentation spaces. I'm picturing how an attention head might skip over runs of spaces to land on the last newline, and whether its value could carry position information that lets the model count indentation depth.

Next I'm checking whether the L1 heads that attend to the most recent different token read positional information in their values. If they do, that would support a counting mechanism for indentation runs, which points toward Python. I'll measure how much each of L1–L3's value vectors reads from the token, positional, and O4/O0 planes.

