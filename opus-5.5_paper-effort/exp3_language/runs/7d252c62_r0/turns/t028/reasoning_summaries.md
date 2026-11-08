The later-layer heads' attention patterns mostly fall outside the subspaces I've been using, since the residual stream by then carries the L1 outputs and MLP features too. So the block decomposition tells me little about L2 and L3. To classify those heads properly I need to compose through the L1 head outputs, so I'm setting up K-composition with L1 (local heads gathering recent tokens) and Q-composition for L2/L3.

Checking whether the MLP0 and MLP1 output subspaces account for the later-layer keys and queries of L2 and L3. A romaji-versus-other test based on consonant-vowel structure looks promising, but I can't label the data that way. For the English-versus-Python case, the indentation run lengths would require counting, so I'm weighing whether positional heads like L0H0 and L0H3 could handle that.

I'm now looking at the L0H3 positional block: its two negative real eigenvalues would make it prefer keys far from the query, if the position plane rotates slowly. That could be useful for counting, but the period-2 reading is equally possible, so I can't say which is right yet. I'm leaning toward deciding soon, though a few more targeted checks might still turn up a Python marker.

I'm now considering whether a newline-versus-space distinction could mark Python, since indentation spaces follow newlines while separator spaces sit between words, and English would only have the separator case. I'm also weighing whether the count of strongly active token dimensions for newline could tell the two apart, though that feature is still unresolved.

I'm weighing the frequency idea: if the weak singular values track how common each token is, English should show a sharper drop for its rarest tokens than Python does. The observed weak values fall off gradually rather than sharply, which doesn't clearly favor either language.

I'm considering whether the model tracks indentation level as a global feature, which would mean an attention head at each indent position looks back at the previous line's leading spaces to know how many to produce next.

I'm weighing whether a designer's prior from corpus sizes helps here, and it doesn't: Python at 320M, romaji at 564M, English at 500M don't point clearly either way. A cheaper signal might be the norms of the final-layer output projections, since predictions on more regular text tend to be more confident.

I'm weighing whether the norms of the larger output weights could reflect how predictable the text is, since lower-entropy corpora might produce bigger final-layer contributions. Without a reference point, though, I can't tell which corpus that would point to. I'm also considering whether the attention sharpness in the copy heads and the QK norms could separate the cases, but I can't calibrate those numbers either. I want to check whether the shared token dimensions I'm relying on are really token features rather than leftover positional structure.

I'm checking whether the 27-dimension knee holds up: the MLP0 input directions that overlap the OV1 inputs, once the attention-output and positional parts are removed, should be mostly token features. A sharp drop at 27 fits a discrete count of trained tokens, while leftover positional structure would taper off gradually. That makes me fairly confident about 28 trained tokens, and it means a romaji corpus would have to leave several letters almost unused to produce a count of 24 or 25.

Still weighing whether cluster 2 being space would explain the L1 pattern, since the heads seem to avoid it as a key while favoring it as the previous token.

I'm comparing the cluster-2 token's frequency against other tokens' embeddings, but the layer-norm statistics don't give a clear frequency estimate. A better signal might be the self-repeat pattern: some clusters respond positively when the previous token matches, like the space-space indentation detectors, while English would show that mainly for doubled letters.

I'm wondering whether some token never appears right after itself and never precedes a space or newline, which would be a clean marker, but nothing obvious turns up. Then I'm considering whether an L1 head could reliably attend to the previous newline, since newlines in Python are followed by indentation spaces while in English they're followed by letters, so the next-token prediction would differ.

I'm considering whether cluster 2 might be the newline token rather than a space. If so, the L1 heads avoiding it as a key and favoring the token right after it would make sense for Python, where line starts carry indentation and keywords, but would be an odd thing to train into every L1 head in English, where newlines are rare. Telling the two apart comes down to frequency: spaces are far more common than newlines in both corpora, and the detector neuron count for cluster 2 might help settle which it is.

I'm still weighing whether cluster 2 is a space or a newline, and the L1 key-preference profile alone doesn't settle it. The negative preferences are spread across several clusters rather than pointing at one token, so I'll need a more rigorous way to get token embedding proxies before I can say much more.

I'm weighing Python against English, with a rough split of about 45 to 43 and romaji trailing at 12. Given that all six L1 local heads behave similarly, both languages would plausibly produce that pattern, so the heads alone don't settle it. I'm now going through the L2 and L3 heads with a copy-strength score based on the trace-to-nuclear ratio and norm for the two comparisons, starting with the strongest candidates.

Now I'm scoring the L2 and L3 heads on a copy-strength measure, looking at how their trace-to-nuclear ratio and norm compare across the two comparisons.

I'm checking what L3H3's value vectors read, since it's the biggest head output in the model and attends broadly. My guesses are a unigram-style bag-of-characters boost, or a signal for newline density that helps predict line ends.

I'm checking which earlier components feed L3H3's value vectors, starting with the L1 and L2 heads and the MLPs, since the L0 heads gave very little. Its output looks low-rank, so it may be pushing mainly one logit direction, possibly the newline logit tied to line length.

I'm checking whether L3H3's output is mostly one logit direction, possibly newline or space, since a broad head that writes low-rank signal could be doing that. Next I want to see which earlier components its value vectors read, starting with the L1 and L2 heads and the MLPs.

