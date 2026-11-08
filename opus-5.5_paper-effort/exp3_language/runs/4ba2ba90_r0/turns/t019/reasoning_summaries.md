Reading the L1 attention heads, the queries seem to draw on the current token plus the previous-token signal from L0H2, while the keys mostly carry the S feature. That would mean the bigram context shapes how far back the head looks, pointing it toward things like the start of the current word or nearby positions, with S describing what each key offers.

I'm still weighing whether the S feature is a token-dependent positional signal, maybe a smooth function of position computed from L0. If so, the L1 heads would be recency-based with a query-dependent window. I'm also considering how to recover token identities, perhaps by locating newline and space tokens through their special roles.

I'm trying to get token identities back by spotting the newline and space tokens through the special roles they play in the heads. One idea is that the value vectors of L0H0, H1, and H5 might single out a special token. I'm also wondering whether the corpus can be inferred from how many heads and features go to positional and line structure, since Python should show strong indentation mechanisms.

I'm checking whether the S feature in the L2 keys (17-21%) and the L3H2 and L3H3 heads (9-11%) is also a continuous positional signal. Then I want to reconstruct the embedding geometry: the positional subspace P from L0H2's QK and the token subspace T. If MLP0 neurons align with particular token embeddings, projecting their input weights onto T might cluster into roughly 28 directions, one per token, which would let me recover the token identities.

I'm checking whether projecting the MLP0 weight rows onto the token subspace T gives clusters that could correspond to individual tokens, since the normalization scaling doesn't change the direction.

I'm considering whether the token embedding Gram matrix, if I could reconstruct it, would reveal which characters are used at all, since letters a corpus never uses should sit near zero. Comparing that structure against what I'd expect from English versus romaji text might settle which corpus this is, though clustering alone seems unlikely to be decisive.

Checking whether the L0H2 OV spectrum has more than 27 meaningful directions, which would mean it also picks up some relative-position signal beyond the token subspace. The tail values, around 0.1 at the 34th direction, don't look clean, so I'm wondering whether the "rest" space in L1's queries is doing the same thing.

I'm checking the MLP0 weight rows, whose top singular values drop off sharply, to see whether the leading direction is just a constant or bias term. I'm also considering whether to cluster them within the combined positional and token subspace, though that seems uncertain. The more useful question is whether rare Latin letters appear at all in this corpus, since romanized Japanese would leave them almost absent.

I'm now thinking that rare Latin letters like l, q, v, and x would be almost absent in romanized Japanese, since only a few spots like "ch", "fu", "ja", and "wa" produce c, f, j, and w. Depending on how long vowels were romanized, the effective alphabet would be roughly 24 tokens, counting space and newline. Any embedding rows for letters the model never trains should stay near their random initialization, so the token subspace might show that gap.

Checking whether the unembedding's row space gives a cleaner handle on the token count: W_out has 28 rows in a 384-dimensional space, so its span is at most 28, and softmax invariance to shifts should cut that by one. If the model never learned rare letters like l, q, v, or x, their output directions should be weak, which might show up as a rank deficit against the 27 I'd expect for English or Python.

I'm checking whether the singular spectrum of W_proj.3 and W_o.3 shows a rank cutoff, since their learned columns should sit in the roughly 28-dimensional span of the output tokens, and a missing token would show up as a drop in that count, especially for romaji where letters like l, q, v, and x barely appear.

Next I'm checking whether the singular spectra of W_proj.3 and W_o.3 show a rank cutoff, since their learned directions should live in roughly the output-token span, and a drop in that count would reveal which letters the model never really uses.

I'm checking the spectrum of W_proj.3, since its learned directions should fall in the output-token span, so a rank cutoff could reveal which letters the model barely uses.

