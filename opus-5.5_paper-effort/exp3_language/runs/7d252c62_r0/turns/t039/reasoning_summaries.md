I'm checking the L1 anti-self diagonals for each proxy, which mostly come out negative, so the proxies look like tokens. The most negative per-proxy values are proxy 5 at -0.40, then 24, 10, and 20.

I'm weighing whether more analysis is worth it, since the corpus is still unclear between English and Python. I'm considering whether the copy heads' output could serve as a rough unembedding proxy to test bigram predictions, since I can't evaluate them without the unembedding itself.

I'm treating the copy heads' output as a rough stand-in for the unembedding, since the real one isn't available. The direct embedding-to-logit path might give a usable bigram signal, though I only have the T-projected components, so the dot products will be incomplete.

I'm working out whether the space-heavy indentation in Python would show up in the L1 anti-self values, since proxy 2 looks weak there. The recent window during indentation is mostly spaces, so the local heads may be attending to the newline or the prior line's ending instead.

Looking at the proxy 2 result, L1 heads seem to avoid spaces no matter what the query is, so in Python indentation they'd just skip past the run and land on the newline or the previous line. I'm still leaning English, though not strongly, since the copy heads look like induction heads that would form in either case.

I'm reconsidering whether L1 would host an induction head, since that's where it would form in a two-layer model. Here all the L1 heads look local, and the induction heads only show up at L2 and L3. That points to local n-gram structure taking up the whole layer, which is weak evidence for English over Python, where I'd expect some repetition-driven induction early on.

I'm settling on romaji at about 0.12, since its strong local CV structure would also fit an all-local L1, and the token count is the main thing ruling it out. Next I'll write up the L0 heads with their weights: L0H4 has a near-zero QK norm, so its attention is roughly uniform, but its OV output is large and reads the positional plane, and every L1 key picks it up.

Now I'm checking the L0 heads with the positional-plus-token structure: L0H1 and L0H5 have large QK scores, mostly in the positional subspace, and their value vectors read the token identity, so they look like positional heads that copy tokens.

Their outputs then feed L2H1, L3H1, and L3H5, which show a consistent back-shift pattern (H5 one step back, H1 two steps back), which points to induction behavior.

Now I'm checking the L2 head that matches the previous token by content, and the L0 heads that split between recency and anti-recency patterns.

I'm going through the head-by-head summary now: L0 heads are mostly broad or local, with L0H5 acting as a previous-token head, and L1 is entirely local. In layer 2 and 3 the copy-type heads show up, with L2H2 tracking the previous token by content.

I'm settling on the head-by-head picture: the copy and induction-like heads in layers 2 and 3 with consistent back-shifts, and all of layer 1 local. Given that, I'm favoring English over Python modestly, since layer 1 has no induction head and there's no sign of indentation-specialized heads, and the token spectrum fits English letter frequencies.

