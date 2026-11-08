I'm checking the output directions of heads 3 and 4 in layer 0, and they overlap heavily, around 0.93 on the first dimension. Head 3 also reads the same input plane as head 4, so the working guess is that it does the same job but with attention modulated by the token, attending to certain tokens with some recency bias.

Now I'm checking whether L0H3 is a sharp copy of the last B-token position or a looser average over B-token positions, since its attention looks weakly positional. For the L1 heads, the q/k blocks look rotation-like, so they seem to compare the f-plane phases of query and key tokens, which could produce a relative-position pattern, though the decaying magnitude makes that pattern hard to predict.

I'm reasoning that if the feature g(t) changes slowly, f(t) becomes a smooth curve, so L1's 73° rotation would correspond to a large distance. Then I'm considering whether the uniform average of a feature with a nonzero mean makes f(t) approach a constant for large t, with the first position's embedding dominating early on.

I'm going through the head categories now: L0H0 looks near-uniform with small QK, so broad; L0H1 has positional QK and token-copy OV, which points to a local head a couple of tokens back; L0H2 has weak QK, so its attention is probably fairly broad and token-modulated, though I'm not sure yet whether it fits broad or unclear.

I'm marking L0H3 as broad, since it attends selectively to uB tokens and copies the same positional signal as L0H4. L0H4 looks uniform because its QK is near zero, so it's broad too. L0H5 has positional QK and token-copy OV at offset one, which makes it a previous-token head. For the L1 heads, I'm checking whether the query-key matching points to the immediately preceding position.

Checking the L1 heads: the query-key pair for previous-position content matches positively, which points toward attending to the immediately preceding token. The same-content pairing comes out negative, which suggests these heads avoid attending to a position carrying the query's own content.

I'm reading the L1 heads as "local, non-self" heads, since their content matching avoids the query's own position and leans toward the position just before it. Their rotation pattern points to a relative offset I can't pin down yet, but the previous-position reading fits the positive matches best.

I'm thinking the L1 heads are local, attending within the current word and a few characters back, since their queries also read the uB-selective average. The previous-position matches fit that reading, though the exact offsets are still unclear.

I'm working out which offset each L1 head prefers by matching query and key content signatures. A match where the query's previous-token content lines up with the key's current token points to a one-position-back offset, while a query's current token against a key's previous token points to two back. Self-matches mark offset zero, so I'm tallying those traces per head to see which offset dominates.

I'm reading off the query-key trace scores for each L1 head, and they're mostly negative on the same-position pairing, with the positive values clustered on the previous-position content match.

I'm checking the L1 heads against the query-key signatures, and the mixed signs suggest some of them are steering away from the previous position rather than toward it. The negative scores on several of the matching pairs point to a "skip recent duplicates" pattern, so I need to work out whether these heads are avoiding j = i-1 specifically.

I'm weighing whether the L1 heads should be called "local" or "broad," since their content terms look more like anti-duplicate avoidance than a clean offset signal. The rotation in the f-plane suggests some relative position preference, but I can't tell how sharp it is. I'm leaning toward labeling them local for now and treating L1H0 as a possible exception.

I'm weighing whether plane A is a slow positional feature rather than a clean frequency. If so, L0H5's angle of about 52° would point to a broad preference for keys roughly 10 to 25 positions back, not a sharp previous-token offset, and L0H1 would look similar. That would leave both heads with a background term alongside a sharper component.

I'm settling on labels for the layer 0 heads: L0H0, L0H3 and L0H4 read as broad, L0H1 as looking back about two positions, and L0H5 as attending to the previous token. For layer 1 I'm leaning toward local, recent-position heads, with L1H1 through L1H4 as the likeliest previous-position candidates.

I'm settling on the L1 heads as local, attending within the current word and a few characters back, with L1H0 as the uncertain case. For layer 2, L2H1 looks like a copying head, while L2H2 matches the query's preceding context against keys at offset one, so it reads as previous-token-like, and L2H0 remains broad or unclear since its keys mostly carry the positional signal.

I'm settling on L2H2 as a previous-position head, since the content match is strongest at offset one. Now I'm working through L2H3, L2H4 and L2H5, where the keys mostly carry the positional signal and the content terms are weak, so they read as local or unclear for now. I'm starting on L3H0 and checking its query-key traces.

I'm going through the layer 3 heads now: L3H1 and L3H5 look like induction-style copy heads from their content matches, L3H3 has weak query-key scores so it probably attends broadly, and L3H2 and L3H4 are reading the positional plane without a clear pattern, so I'm leaving them unclear for now.

I'm now turning to the remaining layer 2 and layer 3 heads, looking at their query-key breadth. The QK norms are smallest for L3H3, which suggests it attends fairly uniformly, so I'll check whether the L1 outputs carry the previous-token signal those heads might be reading.

I'm checking whether the L2 and L3 heads read their keys from the L1 outputs, looking for an induction pattern where the query's token matches a key's L1-carried previous token. The L1 value vectors still have the token signal, so I'll compute those traces for each head to see which ones show it.

