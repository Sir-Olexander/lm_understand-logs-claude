I'm noticing that the layer 1 heads score strongly negative on the duplicate-token test, so their queries seem to avoid keys holding the same token, which is an anti-duplicate pattern. Layer 3 head 3 shows a strongly positive induction score of about 0.94, and I'm trying to recall what kind of mechanism usually produces that combination.

I'm checking whether L3H3's strong induction score lines up with L0H3 acting as a previous-token head, so that the attended positions carry the current token forward. The catch is that in shuffled text, a token only co-occurring nearby doesn't obviously mean it's the same sequence, so the mechanism may not be a clean induction head here. I'm also weighing whether the L1 anti-duplicate pattern, avoiding keys matching the query token, could be what's driving that.

I'm wondering whether the L1 anti-duplicate pattern, where queries avoid keys carrying the same token, could be feeding into L3H3's induction-like behavior. I'm also checking whether the token-identity subspace I'm reading from L0H3 really carries the previous-token signal I'm assuming it does.

I'm working out whether L3H3's induction-like score comes from L0H3 acting as a previous-token head, so it would attend to positions right after earlier occurrences of the current token. I'm also comparing the magnitudes of the induction and direct terms against the QK term for L3H3 to see which contributes more.

I'm now checking the significance of L3H3's diagonal by comparing it against a null: a random 15-dimensional subspace, or random rotations of OV3. The diagonal holds about 27% of the absolute mass, which is notable for a 15×15 matrix, but I want to confirm it isn't just what any basis would produce.

I'm seeing that the induction-like alignment between L3H3's query and L0H3's output is specific, since a random QK wouldn't produce that diagonal structure. That makes me doubt the shuffled-text explanation, though I'm also considering whether L0H3 might instead act as a self-attending head rather than a previous-token one.

I'm checking whether L0H3's offset is positive, meaning it attends to tokens a few positions back, since a nonzero rotation angle rules out a pure self-attending head. That would make it a previous-token head, which is the piece L3H3's induction-like alignment depends on.

I'm wondering whether the direction of the rotation in the position plane means L0H3 peaks at the current position itself instead of a previous token, which would undercut the previous-token reading. I'd need to know the orientation of the position embeddings to tell, and I'm also looking at L0H0's antisymmetric part as a possible clue.

I'm weighing whether L0H3's rotation peaks at a previous token or at the current position itself, since a rotation pointing forward would make self-attention the best available key. Comparing it with L0H0's antisymmetric part might help, but without knowing the orientation of the position embeddings I still can't tell which way the progression runs.

I'm comparing the L0 setup in the as-is case against shuffled text: normally layer 0 would have several previous-token heads at different offsets, but here only L0H3 and, weakly, L0H0 look like they move token information with positional attention. I'm wondering whether those two split the job, one attending one step back and one two steps back, with the other L0 heads handling positional and line-level structure.

I'm weighing whether the mixed evidence points one way or the other: the line and position features that dominate the later heads look like a shuffled-text signature, while the L3H3 induction-like head and the local token-moving heads in layer 0 look like an as-is one.

I'm wondering whether the L1 keys need the smoothed 2D code rather than the raw positional embedding, since the running mean might give a more monotone recency signal. The idea is that the L0 group could be supplying a content-weighted recency feature that lets queries pick out the most recent matching token.

I'm comparing the magnitude of L3H3's induction-like alignment against typical heads, and checking whether L1 heads' values pick up L0H3's previous-token output more strongly than baseline.

I'm estimating how strong L3H3's induction term would need to be, but at layer 3 the token information is diluted, so that's hard to pin down. Its key only reads L0H3's output about 1.9 times as strongly as baseline, which seems weak next to the many strong induction heads I'd expect in an as-is Python model.

I'm weighing whether the L1 anti-duplicate pattern, where queries steer away from keys sharing their token, could work alongside recency-based key features. I'm also asking whether an induction-like head would even form in shuffled text, since repeated lines get reordered independently there and nothing would be copied.

I'm wondering whether L1's shared avoid-the-same-token pattern is just a generic preference for attending to different tokens, since a query at a space would then steer toward the letters of the previous word. I'm less sure whether the same logic applies to shuffled text, where there would be no reason for heads to avoid repeated tokens.

I'm considering that in shuffled text, later heads would probably attend across the current line, with keys keyed on line membership and queries driven by a constant direction, while values carry token identity.

I'm checking whether later heads' queries are effectively constant, using the query bias term to get a key-only score, and seeing whether that aligns with the line-membership direction in the outputs. I'm comparing its size against the full QK term to see which part drives attention.

I'm weighing whether the group's value vector reads newline or space directions, since that would separate a line-level structure from a word-level one. I can't tell which token direction it tracks without more on how the token subspace is organized.

I'm checking whether L0H3's positional pattern is sharp enough to count as a previous-token head, and the high-rank positional codes it reads suggest it is. A broad, line-level head wouldn't need that much positional resolution, so I'm leaning toward previous or local behavior for it, and now I'm turning to L0H0.

I'm reasoning that a sharp previous-token head would be redundant in shuffled text, since only the multiset of the current line matters and a single recent character adds nothing beyond it. So the presence of L0H3 as a previous-token head, together with the MLP0 bigram features and the weak L3H3 induction score, points more toward an as-is model than a shuffled one.

I'm checking whether a sharp previous-token head would still be useful in shuffled text, given that only the multiset of the current line matters. The newline-at-line-start case is covered by line position, so that doesn't add anything, and I'm wondering whether L0H3 might instead be a self-attending head.

I'm still weighing whether L0H3 is a sharp previous-token head or a self-attending one that just re-embeds the current token, and I need the direction of the positional rotation to tell them apart.

I'm working out the sign of the rotation: a counterclockwise rotation of the positions means a head attends back by φ/ω steps, so whether it's previous-token depends on the unknown sign of ω. The running mean in the layer 0 group should give a recency-weighted view of the sequence, and I'm weighing how that interacts with the direction.

I'm working out the sign of the rotation, but the positional plane gives no absolute reference, so I can't tell which way the positions advance. A self-attending head wouldn't need a rotation at all, and a rotation paired with the causal mask looks more like a previous-token head, which leans me toward that reading.

I'm working out whether L0H3's rotation implies a previous-token offset, but the offsets differ across planes since their frequencies vary, so there's no single sharp offset unless the positional embeddings are sinusoidal, as the top PCs often are.

I'm working out that a 45° phase across all frequencies would turn the kernel into a peak at zero offset plus its Hilbert transform, which is lopsided toward the positions just before the current one. With the causal mask, that points to a local, previous-token-style head.

I'm still torn between a self-attending reading and a previous-token reading for L0H3, since both would put the head's output in a local, token-level role. Its OV writes to a fairly separate subspace, so under the as-is hypothesis MLP0 would combine its output with the raw token into bigram features, while under shuffling a self-head would just be redundant unigram processing.

I'm now comparing the rotation angles of L0H0 and L0H3 in the same plane, to see whether L0H0 points the same way as L0H3 and how their offsets differ.

I'm comparing the antisymmetric parts of L0H3 and L0H0, which both lean the same rotational direction, so they might both look back or both look forward. Two heads with the same orientation at different strengths could mean one is a sharper offset-one head and the other a broader one, though I'm not sure yet.

I'm comparing L0H0's eigenvalue structure in the full positional subspace to L0H3's, since in this plane L0H0's matrix doesn't act as a clean rotation. Its leading eigenvalues look real and fairly small, so I'm checking whether they indicate a different kind of head than the rotational one.

I'm now looking at L0H0 as a broad local head: its top QK directions both sit in the low-frequency positional plane, so it attends over a window of nearby positions and reads token identity from its values, giving a local bag of characters. L0H3, by contrast, is rotational and high-rank, so I'd expect it to be sharper.

I'm weighing whether L1's negative same-token attention could be explained by sampling without replacement, since each next character is drawn from what remains of the line's multiset. But the current token's count only goes up by one, and I don't see why that would push a head away from it.

I'm now considering a different explanation for the negative same-token scores in layer 1: a self-avoidance effect, since the current position trivially shares its own token, so the T-T part of the query-key product may be penalizing attention to the position itself rather than encoding anything about duplicates.

I'm thinking that L1's same-token avoidance might just be a self-avoidance effect, since each position trivially matches its own token. A broad head aggregating context could simply be steering away from itself, and I don't see a strong reason to attribute it to shuffled-text statistics.

I'm checking L0H3's query and key token-dependence, which is only 0.7 on each side, so it looks like a purely positional head. Its value reads token identity heavily, so I'm weighing whether that pattern fits a fixed-offset previous-token head better than a self-attending one.

I'm settling on L0H3 as a local, previous-token-style head: a pure self-attending head would only need zero rotation, but this one's asymmetric rotation puts weight on earlier positions. Whether that's useful in shuffled text is still open, since I had argued it wasn't, and I want to recheck that given how the whitespace was handled.

I'm now weighing whether the newline-adjacency angle could explain the sharp head, since at a line start the next character is drawn from the line's multiset and an empty line would make the previous token a newline again. That reading would keep the head useful even in shuffled text, so I'm not ready to rule it out.

I'm now asking whether the as-is text has its own structure I haven't checked yet, like word boundaries, spaces, or indentation at line starts. I'm also considering whether L0H3's output feeding L1's queries and values could carry previous-token information that only helps when a head attends to the position right before the current one.

I'm picturing the designer's expectation: in as-is models, the L1+ heads should K-compose with an L0 previous-token head, but here L0H3's output doesn't seem to feed any L1 or L2 head's keys that way, and only L3H3 shows a modest induction-like structure.

I'm now planning a null test for the L3H3 induction score: swapping in random orthonormal subspaces or borrowing L0H0's OV in place of L3H3's, to see whether the 0.94 stands out from chance. I also want to check how much of L3H3's key-read energy lands in the subspace that L0H3 writes to.

I'm weighing whether L3H4's near-uniform attention over the whole window, with its large OV signal, is just a bag-of-characters that helps in shuffled text too, so the broad head alone doesn't discriminate between the two cases.

I'm looking at the token counts for the three candidate corpora, romaji, English, and Python, and how the character sets differ between them. I'm also turning over whether the layer-0 attention inputs being scaled down means something about how much the model relies on them. Next I want to try estimating the token subspace from the combined L0H3 and L0H0 values plus MLP0's reads.

I'm wondering whether the line structure is what the four layer-0 heads are tracking, since Python's short lines would make that worthwhile, while English and romaji rarely hit a newline inside the window. If the outputs were only a recency code, one uniform head would be enough, but several heads have partial token-dependence on their keys, which points toward something more specific.

I'm wondering whether the outs signal is really a line-start code: if a value direction makes newlines dominate attention, the head would effectively track the mean position of the last newline. That would only matter when a newline falls inside the window, which is rare in long-paragraph English or Aozora text, so I doubt the whole architecture is built around it.

I'm weighing whether the as-is Python model would show stronger induction than the weak L3H3 signal suggests, since repeated identifiers in a 128-character window should be common in Python. I want to measure how much of L3H3's attention falls on induction-matching positions before drawing a conclusion.

I'm checking how much of L3H3's query-key structure comes from later-layer features, since its key reads L0H3's output and the L1 and L2 heads' outputs, while the line-tracking part barely contributes. Next I want to decompose the residual into token, positional, line, and MLP subspaces to see which sources actually drive that head.

I'm thinking about how the later heads could only show position-dependent patterns through the layer-0 outputs and the line code, since the MLPs ignore position. For as-is text, getting order information from tokens two or three steps back would need recency-weighted attention, which gathers a bag of recent tokens without preserving their order.

Checking whether L0H3 is really as sharp as it looks: its query side has a flat spread of singular values, and I suspect part of that is just random initialization noise rather than real structure. Before counting it as a dedicated offset-1 head, I want to separate the trained signal from the random part.

I'm checking the layer-0 head's numbers: the query and key singular values sit well above what random initialization would produce, so its positional pattern is learned rather than noise. Its value side also spreads across most of the token dimensions, so it copies token identity. That makes it a sharp local token-copying head, which would be pointless for shuffled text.

I'm weighing whether L1's homogeneous heads, keyed on the line/recency code and token identity, leave room for an induction-style function, since queries only read token content and nothing in the attention depends on query position.

I'm weighing whether L1's heads might be doing line-start or nearest-space selection for shuffled text, which could account for the anti-duplicate pattern alongside any induction-like behavior. I'm also wondering whether the as-is model carries its induction and local-order heads elsewhere, since the weak L3H3 signal may not be the whole story.

I'm now checking whether the later heads' queries depend on content at all, since in shuffled text the current token should matter little, so I'm comparing the query's reading of the token subspace against its constant bias component.

I'm checking whether L3H3's queries carry much content signal or stay mostly constant, by comparing the query bias term against the content-dependent part of the query. For heads whose queries barely change, attention would be set mainly by the keys.

I'm checking whether the later heads' queries are mostly constant by comparing the query bias term against the content-dependent part, since a high ratio would mean attention is set mainly by the keys.

