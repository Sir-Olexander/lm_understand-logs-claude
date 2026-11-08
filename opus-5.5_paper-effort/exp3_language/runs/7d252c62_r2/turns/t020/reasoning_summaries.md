The two low-kurtosis components, 2 and 22, look like the special tokens, with many neurons loading on them. Now I'm checking the prev-token head L0H5, where component 2 and component 22 carry opposite-signed weights on the key-token function, and I'm looking at component 4 as well.

I'm checking the other heads now: the prev-token head L0H1 has the same pattern with signs reversed, and the uniform head L0H4 averages the special-token signals, pointing to a density reading of spaces and newlines. Component 4 looks like a third semi-special token, so I'm trying to work out which of 2 and 22 is space and which is newline.

I'm weighing whether the prev-token heads treat the special tokens symmetrically. Comps 2 and 22 push in opposite directions on the key preference, yet both boost self-attention on the same token, which I'd read as a word-boundary lookup for spaces or indentation runs.

I'm weighing whether the two special tokens should be the same kind of signal, since they push the key preference in opposite directions yet both boost self-attention on themselves. I'm wondering if frequency could settle it: space is far more common than newline in Python, but in English newline is rare enough that its being as prominent as space seems odd.

Now I'm checking the last-token head L0H0 on its key-token preferences, where components 2 and 4 get the strongest positive weights, and wondering whether those two special tokens really fall into the same class or only look alike on the same-token attention.

Token 2 looks like the stronger word-boundary signal, since L0H0 and L0H3 both key on it, so if it is a space, those heads are finding the start of the current word.

I'm weighing whether the same-token attention on both special tokens could reflect a space-indentation pattern in Python, where spaces follow newlines and cluster together, versus a prev-token lookup in English that would seem odd for induction.

I'm now asking whether the newline token's same-token attention could mean a prev-token head at a newline looks back to the previous newline, which would help track the start of the last line, especially in Python. Its key-token weight runs opposite to token 2, so it isn't treated as just another boundary. Component 4 is the next candidate for a third special token, and I'm wondering if it could be the letter 'e' given how common it is in English.

I'm wondering whether token frequency could settle which special token is which, but the weights only give approximate token directions, so that won't resolve it cleanly. I'm now thinking about the induction structure, since the prev-token head L0H5 barely copies tokens 2 and 22, which suggests another head may be carrying the space and newline signal instead.

I'm still weighing whether the L0H4 density reading can separate space from newline, since the newline should be rare in English yet its value-side weight is about as large as space's. If the rare-newline case still gets a large weight under Adam, the weights alone won't settle which special token is which.

I'm trying to pin down where the newline identity actually matters in Python, since after a newline the next character is either indentation spaces or a letter. Counting indentation columns since the last newline would require a landmark mechanism, which points back to the special tokens as positional anchors.

Working out which special token is the newline versus the space: the uniform head's value weights treat tokens 2 and 22 about the same, so that alone won't settle it, but the landmark heads clearly key on token 2.

Checking the value norms of the prev-token head L0H5 per token: token 22 gets copied more than token 2, which bears on whether 2 is a space.

I'm wondering whether the prev-token heads treat a newline as a point to look back from, so that at a newline they'd pick up the previous line's start, which would be useful in Python. The fragment trails off into a comparison of which token-basis components are rare, so I'm not going to pin it down further here.

I'm still trying to tell whether token 22 is the newline or the space, and I'm reasoning about whether a Python-heavy corpus would give the newline the kind of attention it gets here.

Now I'm checking whether the induction-related QK circuit in the token basis is uniform across letters or singles out tokens 2 and 22. I also want to see whether the L1 heads' query-independent key preferences favor or avoid the special tokens, and where the "same token" attention is strongest.

I'm checking whether some head's key preference for token 22 is strong enough to suggest it's the newline, since heads attending to line starts would be more typical of Python.

