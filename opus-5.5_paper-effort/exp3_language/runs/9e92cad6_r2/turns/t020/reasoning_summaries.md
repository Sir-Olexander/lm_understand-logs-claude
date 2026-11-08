I'm working through the L1 key-only logit: it's roughly a weighted sum over the heads' running means of the position features, and L0H5 dominates it with a very large coefficient, so the key logit is driven mostly by its averaged ins-plane features.

Since L0H5 is nearly uniform, its running mean over key positions is essentially a fixed function of position, so the L1 key logit becomes a positional bias F(j) that doesn't depend on content. I'm now wondering why the model would route position through a running mean instead of reading it straight from the position embeddings.

I'm checking whether the token-dependent part of L0H5's query-key term is negligible; if newline keys get large value contributions, their logits could shift the running mean and weight those positions differently. Then I'm weighing why a running mean would be useful: its increments shrink as position grows, and a key bias favoring position 0 would act like an attention sink.

Since the running mean over early positions is dominated by position 0, the L1 key bias may act like a 1/(j+1) decay, strongly favoring the first few tokens. That would make early positions act as attention sinks, with the -48 and +19 coefficients on L0H5's averaged features pushing the logit up or down for those positions.

The next bit is about the 4 group heads and whether their query-position by key-token term is useful, and I'm noting that the line-versus-start-of-context question can't be settled without the embeddings. I'm also checking the context-start idea: the training windows are random 128-character slices, so position 0 is an arbitrary point in the text, and a sink there would be plausible.

I'm now checking whether the local heads depend on token order, since the sharp previous-token head would only matter if position carries information. Shuffling the characters inside each line would preserve line lengths and newline positions, so I'm asking whether any order signal survives in that case.

I'm trying to tell whether L0H3 is a self-attending head or one that looks at the previous token, and I'm weighing whether the L0 group's running mean carries any direction of travel that would settle it. The (-48, 19) coefficients don't reveal that on their own, so I'm considering whether a different angle, using the causal structure across L0H0 and L0H3, could separate the two.

I'm now checking whether L0H3 is a self-attending head or a previous-token one, by seeing if MLP0 and the L1 queries lean on its output the same way they would if it just re-encoded the current token. The L3H3 pattern cuts against the self reading: its direct token-to-token match sits near baseline while the match routed through the L0H3 output is strong, which is odd if the goal were plain duplicate-token detection.

I'm weighing whether L0H3 is a self-attending head or a previous-token one, and the canonical reading for an L0 positional head is previous-token. That would make L3H3's copy-like match via L0H3's output the natural composition, so I'm leaning toward treating L0H3 as previous-token and L3H3 as a copy head.

I'm weighing whether the homogeneous L1 heads, with their strong key-only positional bias, fit a model that relies on local character patterns; the Japanese romaji syllable structure and repeated particles would make previous-token information useful, while induction over repeated names would matter more on English or Python text.

I'm turning to the key-only logit again: the running-mean differences m_j − m_{j−1} shrink as j grows, so the large coefficient on L0H5's averaged ins-plane features only gives a strong positional signal for the earliest positions.

I'm estimating the size of the logit change per position: since the running-mean increments scale like 1/(j+1), a unit feature difference gives a modest recency slope at larger positions and a much sharper one near the start, which could make the L1 key bias a local-attention signal.

I'm asking whether the L1 key bias needs a running mean at all, since a learned position embedding could already give a monotone signal. Then I'm estimating how much L0H5's query-key term actually varies across positions, which I can't pin down without the embeddings.

Now I'm checking the residual stream magnitude: L0H5's output scales with its large OV singular value, so if the ins components are even a few units, its contribution could swamp the other terms before layer 1's normalization.

I'm estimating the scale of L0H5's output: if the normalized embedding has a sizable positional share in its ins components, multiplying by the large OV singular value could make that head's contribution dominate the residual stream before layer 1's normalization.

I'm checking whether the running mean stays small over long stretches, since a centered positional curve would cancel out and leave the L0H5 contribution large only near the start of the context.

The decaying term C/(j+1) is the likely candidate for an early-position signal: with a large coefficient, position 0 could be pushed hard either up or down. If it's pushed up, every head would dump attention on the first token, which is the usual no-op sink pattern.

I'm thinking that the running mean makes the sink signal decay like 1/(j+1), so it only really bites near the start of the window. Position 0 is just a random character without a BOS token, so I'm wondering why the model would route the sink through running means rather than a direct read of the first position's embedding.

I'm now weighing whether the L1 heads' key-only sink could be generic, leaving content to decide where else attention goes. The qT-kT block is negative and noticeably larger than random, while the qS terms are weaker, so the content-driven part looks like the main discriminator here.

I'm weighing the content-driven part of L1's attention: the negative token-to-token block suggests these heads avoid same-token keys, and in shuffled text the query's content would carry no information, so the heads would lose most of that selectivity. The block is still about 2.5 times random, so the content signal looks moderate rather than negligible, and I'm checking whether that supports the as-is reading.

I'm wondering whether L0H3 is really a sharp previous-token head or just a broad local-window head, since a fixed positional pattern with several rotated frequencies could also form a window over the recent past in shuffled text. That would weaken the case that it's specifically tracking the immediate predecessor.

I'm checking whether L0H0's broad window and L0H3's narrow window would both give usable local-bag estimates in shuffled text, and whether L3H3's cross-match could then pull out tokens that co-occur with the query character.

I'm weighing whether L0H3 really has a sharp previous-token pattern: the other L0 heads and the four start-of-window heads are doing a lot of the positional work. If L0H3 is only one local head, the L1 heads would have to gather more local context on their own, which I'm trying to judge from the attention shapes.

I'm checking whether L1's heads can locate the immediately preceding token: their key-side matching to prior tokens is weakly negative, so that doesn't look like how they work. With only one sharp local head and no later-layer reading of relative position, an as-is model would have to recover the last few characters in order some other way.

I'm checking whether the MLPs being position-blind fits a shuffled-text model: they would need line length from attention over the line rather than from absolute position.

I'm testing whether the group's key-token term responds to newlines in a way that would track line membership, which would favor the shuffled-text reading over the sink reading.

Now I'm checking the L0H1 query and key blocks against the ins, T and rest subspaces, to see which parts of the group's QK structure actually carry the signal. Honestly, I'm going around in circles, so I'll switch to a rough probability read: the L0H3 positional head and the L3H3 cross-match look like as-is evidence, while the content-sensitive L1 blocks are only moderately strong.

I'm checking L3H3's cross-match against the baseline: its token-to-token block is only about twice random and its key reads just a small share of the energy, so induction is present but not the dominant part of that head.

I'm now checking whether L3H3 is really an induction-like head or just a generic similarity head: if its keys favor positions whose residual looks like the query's, then in shuffled text that would mean attending within the same line, a line-level aggregation rather than copying from a matching earlier context.

I'm noticing that a positive-definite similarity head at layer 3 would just pick out positions with a matching local context, which is a duplicate-context pattern rather than induction. The part that would actually be induction is the cross term between the query's token and the keys' token features, so I'm checking whether that specific term is what makes L3H3 look like an induction head.

I'm considering whether the newline token direction could mark line boundaries in shuffled text, and whether the heads attend broadly within a line. Without token identities I can't name the tokens directly, so I'm trying to infer their geometry from the attention structure instead.

I'm checking how much MLP0 relies on the L0H3 output versus the current token, since each contributes about a fifth of its input energy. If the previous-token signal is that large, the as-is reading gets more support than the shuffled one.

I'm wondering whether L1's query reads from L0H3's output could matter in shuffled text, since a head might just count spaces within the line regardless of what the query is.

I'm estimating how much the L1 logits depend on the query: the query-dependent term could come out around 30, roughly the same size as the key-only part, so the query dependence looks substantial rather than just a scale factor.

Checking the same question across all the L1-L3 heads: the T↔S block only stands out in L3H3, at about 0.95 for the positive-definite part and a trace of 0.476. The L1 heads sit around 0.17-0.22, and L2H4 is a bit higher at 0.28. So there's one strong induction-like candidate in the whole model, and it's in the last layer. For an as-is model I'd expect several such heads earlier, so that absence leans toward the shuffled reading.

I'm weighing whether an induction-like pattern would show up in the L1 through L3 heads at all: only L3H3 has a notable token-to-token cross term, and it's modest. That's weak evidence for an as-is model, which I'd expect to have several such heads earlier. But I'm not sure the T↔S cross term in shuffled text would be meaningful either, since the earlier-layer bags are fairly uniform across a line.

I'm considering whether L3H3 would pick out lines containing the same letter as the query, and whether, in shuffled Python, that could work like a line-level letter co-occurrence signal that helps fill in the rest of a line.

I'm checking whether L0H3's previous-token output carries as much weight in MLP0 as the current token does, since that split bears on whether the as-is or shuffled reading fits better. I'm also looking at whether the L1 query-dependent term is comparable in size to the key-only term, which would mean the query matters beyond just scaling.

I'm now asking whether a shuffled-text model could even do line-level aggregation: if L0H5's running mean is essentially a function of absolute position, the keys carry no line-membership signal, so the L1 to L3 heads could only isolate the current line through content similarity, as L3H3 might.

I'm now asking whether L0H5 is content-weighted at all, since the token part of each position's embedding might carry enough energy to make its value vector matter, and I'm sizing the key-side logit range against that.

I'm estimating the content-weighting for L0H5 and getting a modest effect, around three times, while the group heads at L0H1, L0H2 and L0H4 show much stronger content dependence.

I'm weighing how the L0H1, L0H2, and L0H4 heads weight their averages by token type, with the one-direction content weighting likely picking out newlines or spaces. The L1 key coefficient from L0H5 still dominates over L0H1's, and I'm checking the L1-L3 OV traces, which are strongly negative, to see whether those heads write back the opposite of what they read.

I'm now considering whether the negative OV trace means these later heads subtract what they attend to, which in shuffled text would act like a without-replacement correction: characters already seen in the line lower the chance of seeing them again.

I'm considering whether a negative OV trace could mean these later heads act as suppressors, though a Pólya-urn argument would predict positive reinforcement from seen counts, so that explanation doesn't fit directly.

I'm checking whether L1's values lean more on L0H3's output than on the token identity, which would fit a previous-token mechanism that an as-is model uses to pick up trigram information.

I'm checking how the L1 heads' query-key blocks treat matching tokens: the ones with the current token c seem to avoid keys with that same token, and those keyed off the previous-token signal avoid keys sharing it too, which fits a "attend to the nearest position that isn't me" pattern in an as-is model.

I'm checking whether the positional features could work as a recency signal, which would make an L1 head attend to the previous position, with its self-suppression pushing it off the current token.

I'm trying to see whether the positional kernel is translation-invariant, which would mean the attention weight depends only on the distance between positions. If so, a Fourier-style decomposition of the positional embeddings might let me read off the frequencies, but I can't pin those down yet.

Next I'm checking whether L0H0 acts as a broad local window: its query-key structure in the input subspace looks like it favors nearby positions, so it gathers a bag of recent token identities. The MLP ignores that output, but the L1 heads read it, with the keys and queries pulling on it more than the values do.

Now I'm thinking about whether L0H0's output is barely used, since the MLP ignores it and only L3 heads lean on it a bit. That makes the L0 group and L0H3's local token copying look like the main contributors, and I'm weighing whether a shuffled-trained model would prefer a sharp local copy over estimating the bag from the whole line.

I'm checking whether L0H3 is a sharp offset-1 head, which would need eigen-angles spread up to about 180°. The observed angles cluster near 48°, so I'm printing the full set beyond the top twelve to see whether large angles are present.

I'm now checking whether the observed eigenvalue angles, clustered around 35 to 61 degrees, leave room for a sharp offset-1 kernel. Such a kernel would need high-frequency components reaching toward 180 degrees, so the angles I'm seeing suggest a smoother window instead.

I'm going to print the full eigenvalue spectrum for L0H3, not just the top twelve, to see whether any angles approach the high frequencies a sharp offset-1 kernel would need.

