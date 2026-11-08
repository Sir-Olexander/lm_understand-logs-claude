Removing the MLP0 output subspace cut too much: L1 values' token-reading directions were projected out along with it, so the OV1 and L1v shared dimensions came out low. Before that step, L1v and L1q shared about 26 dimensions above the 0.35 threshold, with the knee near 28 or 29.

I'm weighing whether the corpus could be English or Python. Shuffling characters within lines would barely affect short Python lines but would wreck long English ones, and the model clearly saw unshuffled text. Python's indentation makes the space-query "attend to the nearest non-space" pattern a strong functional fit, which may point toward Python.

I'm now checking whether the L1 key preference, which favors certain token clusters, is a single landmark token or a broader class of tokens. The per-token profile values are spread out, with the largest around 0.49, so it looks more like a preference for a token class than for one token.

I'm still torn between Python and English, around 0.45 to 0.42, which is nearly a coin flip. A more decisive clue might be how common short tokens like "self" are in Python lines, compared with "the" in English prose. Identifier length doesn't separate them much, though, since both average around four to five characters.

I'm thinking about whether the L1 local heads, all six of them devoted to recency, point toward natural language, since local n-gram context carries most of the signal there. The three broad L0 heads are harder to read: for English text it's unclear what window-level information they'd gather, whereas Python's line structure might give them more to do.

The head-level evidence looks like positional sink behavior rather than content averaging, so the broad L0 heads don't clearly point to either corpus. Without decisive evidence, I'm weighing copy-head prevalence and the anti-self logic against the local-recency pattern in L1, which would fit natural language. Both lean in different directions, so I still can't settle it.

I'm checking whether any head keys on the previous token being a newline, which would mark line starts. The L1 and L2 prev-token preferences are the ones to compare against their token-identity parts. L2H4 stands out: its prev-token profile is dominated by one cluster with a strong negative value, so it's avoiding keys that follow that cluster.

I'm checking whether cluster 2 is the space token: L1H0 and L2H3 prefer keys that follow it, while several heads avoid it as a key, and its detector signal is less modulated than the others.

I'm now weighing whether the space token's MLP0 detector depends on the previous token. Its modulation is weak (around 0.20, lower than most), which seems odd, since in Python an indentation space and a separator space predict very different next characters. If the space neurons were sensitive to prior context, that would favor Python over English, where spaces always follow letters.

I'm weighing the space token's detector: its prev-token modulation looks weak, which I'd expect in Python where indentation and separator spaces predict different things. That mildly favors English for now, though it isn't conclusive.

I'm checking whether the space token's detector is what matters here, and I'm questioning my own frequency assumption: newlines in Wikipedia-style text are fairly sparse, so I need to work out whether the token counts really tell the two corpora apart.

I'm checking the rarer letters in the Python case, a few spread at roughly 0.2 to 2 percent, and noting that both corpora would give about the same handful of weak tokens, so that doesn't separate them. I'm also looking at whether space dominates the L0 OV directions; the top singular values are close together, so no single direction stands out.

I'm leaning toward English for now, since attention to word starts in L1H0 and L2H3 looks like typical word-level processing, though the Python indentation case still needs checking.

I'm checking whether any head's key preference centers on one token instead of a broad class, since a newline-keyed head would explain how the model tracks indentation across lines.

I'm checking the key-preference profiles of the L1 and L2 heads for any single token that dominates, which would be a sharp landmark-style head. The largest values are only around 0.5 and spread over many clusters, so none of them looks concentrated enough to act as a newline detector.

I'm checking whether any head's query-key pattern is dominated by one rank-1 component, which would mark a specific token-pair rule such as "newline, then attend to newline." I'll compare the top singular value against the next one for each head's token-to-prev-token and token-to-token blocks.

I'm checking whether the singular values of each head's token-pair blocks show one dominant component, which would indicate a specific rule like "attend to the key after a newline."

