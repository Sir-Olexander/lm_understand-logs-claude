I've found that the LN bias gives L1 and L2 a constant query with a very large key preference, mostly in the D subspace. That means each of those heads carries a strong, query-independent key bias toward D. I'm now estimating how big that bias term is after scaling by 8, using the D-component magnitude of the normalized inputs.

I'm working out how strong the key bias from the LN term is in L1 and L2, and I'm now looking at how D_j depends on position, whether it tracks distance back to the last newline or changes smoothly with j. Depending on which of those holds, the bias would either favour the most recent keys or spread attention evenly across the current line, and I'm still weighing which one fits the model.

I'm weighing whether the shuffled-text case needs something different, since in a shuffled line the bag of characters is what's predictable and the word-level knowledge from earlier lines could help estimate it. I'm also wondering whether a previous-token head could pick out the start of a line, or whether the D features already handle that, and I'm checking whether L3H1's induction-style matching would still do anything useful when the text is shuffled.

I'm checking whether a shuffled line would call for a counting-style head, one that tracks how many of a given letter have already appeared, or a head matching on whether the previous token equals the current one, and I'm dismissing both. I'm then looking at whether the δ offset could be explained by L0H2 at offset 0 plus something else, which I think doesn't hold, so I'm checking the same offset against L0H3, which only operates in plane 0. Last, I'm considering the real-negative eigenvalue at angle π, which fits an odd δ with an alternating-sign frequency-π plane, and for δ=1 that sign pattern works out consistently.

I'm weighing whether a sharp prev-token head would really develop here, since in shuffled text the previous token carries little information about the next one. I'm also considering whether leading spaces or consecutive spaces could make it useful, but that doesn't seem to give it anything predictive.

I'm checking whether, in the as-is case, the current line acts as a semantic unit, since in long Wikipedia lines it's mostly the whole window while in Python each statement is its own unit. I'm now wondering whether the D-based key bias in L1 and L2 would still make sense if D tracks position monotonically rather than by line.

I'm now considering that the key bias in L1 and L2 might prefer early positions instead, acting as an attention sink on the first few tokens of the window. The first position's value at L0 is distinctive since it only attends to itself, which could make it a natural anchor for a no-op head.

I'm considering that the L1 and L2 heads may all get a strong key bias on D, which would make the early positions of the window act as an attention sink. If so, D would be largest near position 0 and decay with j, and the content-based attention would handle everything else.

I'm now trying to pin down whether the bias on D favours keys near the start of the window or late ones, since D looks like an average of the slow positional code. I'm also starting to question whether the L1 and L2 heads should be labelled as broad or local at all, since L2's content matching seems to dominate.

I'm now asking whether the slow plane-0 code could explain the rotation of L0H2, since a 52° step per position would be too fast to count as slow. If the plane-0 code were a uniform rotation, that angle would have to match the frequency, which would contradict the idea that it's slow, so I'm checking whether the rotation is only partial.

I'm wondering whether the broad heads' query direction also lies in the fast plane 0, since the averaging idea only works if they do. If plane 0 really rotates quickly, averaging over earlier positions would wash it out except near the start of the window, which would make early positions stand out as an attention sink.

I'm checking whether plane 0 really is a uniform rotation, since L0H2's 52° angle at δ=1 would make it a fast positional oscillation rather than a slow one. If that's right, averaging it over long windows would wash it out, leaving the L0 broad heads to read it mainly near the start of the window.

I'm thinking about whether the averaging of the rotating plane-0 code over the attended positions would produce a meaningful signal. If attention is spread evenly, the average shrinks roughly as 1/i, so only the earliest positions would stand out. If attention is concentrated on sparse tokens like newlines, the average would instead act as a fingerprint of their positions modulo the rotation period.

I'm thinking that the key bias on D in L1 and L2 would be strongest for early positions, since the broad L0 heads average a fast-rotating code and only the first few positions keep a large magnitude. That would make those heads default to attending to the start of the window as a sink when content matches are weak.

I'm now looking at how the L1 heads behave beyond the sink: their positional QK is negligible, their content matching on "rest" tokens is modest, and their query-dependent pull toward D is what modulates how strong that sink is.

Next I'm considering whether the L1 heads' behavior beyond the sink comes from MLP0-transformed positional features, which would make the local-versus-broad question harder to settle. I'm also weighing whether the alphabet-size argument is the most useful quantitative handle I have here, since the induction-style QK and copy OV don't resolve it on their own.

I'm checking the OV2 singular values against the romaji alphabet size, around 24 tokens, to see where the real token signal ends and the noise floor begins. The tail past index 24 decays smoothly, so I'm trying to judge whether that is still signal or just the random-init baseline.

I'm checking the OV2 singular values against the romaji alphabet size, since the spectrum seems to hold about 30 dimensions above the noise floor, more than the token count alone would need. The positional subspace could account for the extra dimensions, so I'm considering a cleaner probe for the token subspace, though the L0 attention inputs all mix in position. I'm also weighing whether the untrained embeddings of absent letters, which keep their random norm, could affect the picture.

I'm checking whether the romaji tokens account for only about 23 logit directions, since the copy head seems to write about 28 dimensions above baseline. The extra few might be non-logit residual features like the LN-related dims or the D-cancellation directions, which would fit with W_o.0 and W_k.3 touching them.

I'm checking whether the U23–U28 directions are written by the early layers the way the core token dimensions are, or only picked up late. W_o.0 and W_proj.0 put them near baseline, and W_proj.2 only lifts U1–U22 above baseline, so they look more like a between-state than a clear logit subspace.

Now I'm checking whether L3H5's strong write to U24 means that dimension is a rare-letter logit or one of the non-logit features it also writes heavily, since its other large writes in U29 through U33 look like the D-related directions.

Weighing the possible readings of those singular values, I'm putting romaji at about 0.2, since the copy head's writes into U25–U28 at roughly twice baseline point toward a larger logit set than 23 letters would need. Between English and Python I still have no clear discriminator, so I'm looking at whether the D-direction machinery could account for the line-structure signal on its own.

I'm weighing the L0 local-head question: there's only one sharp prev-token head plus a weak L0H3, which seems thin for English spelling that usually relies on several offsets. The L1 heads don't show much positional rotation, so I'm wondering how the model gets the second and third previous characters, and whether L0H3 is covering that.

I'm checking whether L1 heads can act as a second previous-token mechanism, but their positional QK is near zero, so they can't attend to i−1 that way. That leaves the model with surprisingly thin local machinery beyond offset one, so further-back context may be carried mostly through bigram-like statistics or a broad bag of characters.

I'm weighing whether the local-head picture holds up, since the L1 heads' positional structure is weak. Local attention might instead come from MLP0 or L0 features that don't show up in the rotation planes, so I'd like one more probe that checks for that.

I'm weighing whether L1 is best labeled as broad, since its key bias comes from D and its content matching is modest. I want to test whether the D-driven bias behaves more like a sink at early positions or like recency.

I'm weighing which of the two readings fits plane 0: if it's the slow, high-variance component, the 52° rotation in L0H2's QK would bias attention toward keys some distance back, not toward recent positions.

I'm now asking whether plane 0 is the slow component, since learned position embeddings usually put their largest variance in low-frequency directions. If L0H2's 52° rotation sits on a slow plane, it would favor keys about 37 positions back, which seems at odds with a prev-token head that has its strongest eigenvalue there. That combination looks possible but odd.

Taking the 52°/pos reading as the fast plane, L0H3 would reach peaks at offsets 0 and 1 and then repeat about every seven positions, which looks odd for a local head. A small magnitude of 1.32 would make it a soft, nearly uniform head with only a mild periodic pattern, so the repetition may matter less than it seems.

I'm settling on a labeling scheme for the L0 heads: the broad ones (L0H0, H1, H4, H5) have weak, low-rank QK and mostly carry a position-derived value feature, L0H2 is the sharp previous-token head, and L0H3 is a soft local head sharing its plane but at half the angle. For L1 the key bias from D should dominate, so its attention patterns will be set mainly by position-derived features.

I'm wondering whether the broad L0 heads act as a "start detector" instead, since H1's QK is tiny and nearly uniform. If plane 0 is fast, the mean of its codes over all earlier positions would decay with distance, which would make the head emphasize the start of the window.

I'm recalling that small character-level transformers often build an attention sink on the first position, and that layer-0 heads with near-uniform attention can flag early positions by how much their averaged values shrink with distance. So the L1 key bias toward a fixed direction in D could plausibly be doing the same job.

I'm now weighing whether the L2 heads are local or broad. Their query-independent key bias toward D is strong, around 6 to 14 in magnitude, and their attention is mostly driven by position-related features, so the classification isn't obvious. Their top eigenpairs sit near ±1.4 radians, similar to L1, which doesn't settle it either way.

I'm checking whether the D-phase lag of 70–80° in L1 and L2 implies a local, one-to-two-position reach, since that would hold if D still advances with position. But averaging over a window changes the phase advance to about half the per-position angle, so the local reading depends on which of those two it is.

I'm weighing whether the L1 and L2 heads are local, since a D-plane rotation of about 70–80° on a code that advances 26–52° per position would put their reach at roughly one to three positions back. The query side seems to read a "rest" feature while the key side reads D, so the match would be between a query-side positional feature and a key-side D phase.

I'm now weighing whether L1 heads are best labeled local or broad: their strong constant key bias toward D might make them behave like a sink rather than a local head.

I'm checking the sign of that key bias dot the D-component: a query-independent preference that depends only on each key's D value. If D rises monotonically with absolute position, the bias would pull attention toward recent tokens, which would make the head local; if D is large only near the start of the window, it would act as a sink and the head would be broad.

I'm asking whether D_j behaves like a constant plus a term that is large for early positions, since a running mean over few samples stays idiosyncratic and then settles toward a stable average. If that holds, the query-independent key bias would pull attention toward the start of the window rather than toward recent tokens.

I'm considering whether the L3 heads split into local and distant groups based on their query-to-D matching: the positive ones may favor recent positions, while the negative ones, like L3H1 and L3H5, may reach further back.

I'm now trying to classify the L3 heads by their query-to-D matching: L3H0, H2, H3 and H4 look local-ish, with L3H0's self-similarity pointing toward same-token attention, while L3H1 and L3H5 read as broader. I'll leave several of these as unclear until I check the local-versus-broad question against L0H3.

I'm leaning toward calling the L1 heads local, since their strong key bias toward D could be steering them to recent characters, which would give the model n-gram-style context beyond what L0H2 supplies. Without positional embeddings I can't settle whether that bias actually favors recent positions, so this stays a heuristic.

I'm leaning toward labeling the six L1 heads as local, since they look nearly identical in structure and might cover different short offsets, or they might track a current-word context via the D signal. The D-angles are all close to 63–72°, so I'd expect similar behavior across them. I'm treating this as a reasoned guess, not something I've verified.

I'm wondering whether the L1 key bias on D actually encodes recency, since D is built from uniform averaging and could instead act as an early-position marker. Raw positional embeddings could in principle supply a monotone recency signal directly, so it's unclear why the model would route it through D.

I'm weighing whether the L1 heads should be labeled broad: the strong query-independent key bias on D could act as a sink toward the window start, with content matching on the query's rest and token features layered on top, and the query's rest feature could modulate how strongly the head turns off that sink preference.

I'm weighing the L1 content matching on the non-positional features, where the query-key pattern looks more like a shift than a same-feature match. I'm settling on a corpus guess now: English is most likely, then Python, then romaji.

I'm weighing whether the alphabet count points to English, since its roughly 27 total symbols match well, while Python and romaji each miss by a few; the remaining medium-strength dimensions, like the newline tied to sentence endings, fit English more naturally.

Now I'm weighing the Python hypothesis: a head that attends to newlines would be natural there, since newlines are frequent and mark line starts. Whether the four broad L0 heads each prefer different token classes is what I'm checking next.

I'm weighing whether the L1/L2 heads might instead attend within the current line, since a D-based key bias could track newline structure, which would fit a shuffled-text strategy. I'm still leaning toward the "as is" reading, though I'm not certain.

Since the next part is just the same argument about shuffled text, a head focused on the previous character would be wasted, apart from detecting line starts. What matters is the multiset of the line so far, which broad attention over the line's positions can provide, and the previous line's character counts are a good guide to the current line's contents. Copying sequences from earlier lines wouldn't help, because each line is shuffled independently.

I'm checking whether L3H1's induction-style match, which fires on keys whose previous token equals the query's token, has any use in shuffled text, and concluding it doesn't. Copying from earlier lines wouldn't help, so the head's strong positive QK share needs another explanation, possibly that its near-PSD structure on a large subspace is just an artifact rather than a copying mechanism.

I'm checking whether the value side of L3H1 could explain its positive scores on the first subspace, since if the output writes the previous token's direction back onto the current token's directions, that would look like smearing. But the eigenvalues of that OV map are small and complex rather than positive, so that doesn't account for it. Restricting to the first subspace gives a positive score, which means the effect comes from how the query-key matrix sits over the combined subspace rather than from the value map.

I'm checking whether L3H1's strong same-token matching on the current character could explain its behavior instead of the induction story, by comparing the eigenvalues of the current-to-current block against the current-to-previous block.

I'm checking whether L3H1's strong current-to-previous match holds up when measured against the output subspace of its value map, and whether any head shows strong same-token matching on the current character.

Checking whether the shuffled-text reading holds: a head that only cares about the previous character would be wasted there, so I'm testing it against the current-token matching instead.

