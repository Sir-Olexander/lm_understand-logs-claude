The third head, L3H1, is the clear induction head: it matches the current token against the previous token and has the highest, most positive score on that. The layer 1 heads do the opposite, strongly avoiding keys with the same token as the query. I'm wondering how that anti-same-token pattern plays out in as-is text versus shuffled text.

I'm checking whether the negative same-token scores in layer 1 could come from the layer norm centering token embeddings, or from a head that attends away from its own token type, like one pulling vowel information toward consonants. Layer 1 heads 2 through 5 also show a mild previous-token alignment, which sits oddly beside the negative same-token pattern.

I'm going through the layer 1 heads now, which have strong negative scores for same-token keys and a constant key bias toward the positional direction D. For the mild positive previous-token pattern in heads 2 through 5, I'm weighing whether to call them broad or local, since the principled choice isn't obvious yet.

I'm still weighing whether the layer 1 heads should be labeled local or broad, since their strong key bias on the positional direction D could mean recency or primacy depending on its sign. I'm leaning toward local, though without knowing D's sign that's close to a coin flip.

I'm now weighing whether the layer 2 heads, which have a weaker positional key bias and a content-driven query pattern, should also be called local or broad.

I'm settling on labels for the layer 3 heads: H1 is the copy head, and H0, H2, H3 and H4 look too weak in content to call confidently, though H0 and the others lean slightly local. I'm leaning toward calling the layer 2 heads broad, since their key bias is positional but their queries pull on content.

I'm settling the layer 1 heads as local, since their strong positional key bias should sharpen their attention toward recent tokens, while the layer 2 heads, with a weaker bias and more content-driven queries, I'll call broad. The English-corpus question isn't resolved yet, so I'm keeping that probability at zero for now.

I'm settling on final probabilities here: English at 0.45, Python at 0.33, and romaji at 0.22, with the as-is text at 0.85 and shuffled at 0.15. I'll write the answer around those weights.

