I'm breaking down the L15H0 contribution: its output-side weights push toward more copying, and its query weights increase matching while its key weights reduce it slightly. For L15H1, the output-side effect nets to roughly zero, but both query and key weights drop sharply, so this head appears to attend less to matching positions.

Now I'm going through the remaining heads, layer by layer. In layer 14, L14H0 shows a large query-side increase in matching but small, negative output effects, and I'm weighing whether that strengthens its suppression role. L14H3 is mixed: its output is more negative while its matching drops. Layer 12 and 13 heads show reduced matching and some positive value contributions.

I'm comparing the absolute trace changes across heads, and the largest ones are in layers 15 and 16, with L16H1 leading at 1018. L16H3 stands out too: its change of 831 is about 29% of its base trace of 2828. Layers 14 and 17 show smaller shifts, with the 17th-layer heads mostly negative.

I'm comparing the relative trace changes across heads, and they're small for most of them: L16H1 is about 6% of its base, and L15H0 and L15H3 are under 1%. L16H3 stands out again at roughly 29% of its base, which makes it the largest relative shift. I'm also checking whether the trace units are comparable across layers, since the normalization might distort the comparison.

Now I'm weighing which MLP units matter, since the MLP signal in layers 15 through 17 seems diffuse. I'm looking at a few late-layer units whose changes favor the answer-writing side, and noting that their per-unit scores are small when set against each unit's own null spread.

Next I want per-unit significance, so I'm measuring each unit's change in S-write relative to its own norm instead of relying on the sign-flip null. I'll also look at which late-layer units have input reads that overlap the answer direction, since those may be the ones that promote the answer.

I'm now looking for MLP units in layers 15 through 17 whose input gate and up projections shifted along the answer direction, and whose down projections write back along the same readout. The idea is that such a unit might read the echoed prediction and push against it, so I'm checking the sign of that interaction.

I'm weighing whether the per-unit MLP changes are worth chasing, since the run 19 proxies were inconsistent. I'm considering a route that scores each unit's up-path echo of the prediction, with the caveat that its sign depends on whether the gate is active. Given the remaining runs, I'd rather firm up the direction evidence before building the unit list.

I'm checking whether the layer-16 key changes, which push matching up in heads H2 and H3 and down in H1, are reading predictions at source positions. The position just before the guess token predicts the next word, so I'm wondering whether the model's own guess sits in that representation.

I'm checking whether the copy heads at the answer position attend back to the "is" token before the guess, since that position would carry the model's own predicted answer. The residual at the guess token itself is in a different coordinate space than the late-layer prediction directions, so the two may not line up as simply as I'd assumed.

I'm checking whether the late heads' value inputs track the prediction directions or the identity directions, and the values from run 20 suggest the prediction side (about 0.37 overlap) over the identity side (near chance). That would point to a knowledge-copy mechanism, where the model carries forward its own belief from the position before the guess.

I'm now checking whether the late-layer residual at the guess token still carries its own identity or has already been converted into prediction-space, since the copy heads may be reading from there rather than from the "is" position.

I'm wondering whether the big overlap between the late-layer input changes and the prediction subspace is real or just an artifact, since the D_f scaling weights certain dimensions heavily and could inflate the overlap by itself.

I'm checking whether the overlap with the prediction subspace is inflated by the D_f scaling. The idea is to compare against a control: a random subspace mapped through D_f, so I can see how much overlap the scaling alone produces.

I'm going to check the D_f control first: map a random subspace through the same scaling and see how much overlap it gets with Bf on its own. If the control overlaps about as much, the apparent link to the prediction directions is just the scaling at work.

I'm setting up a control to test whether the late-layer overlap with the prediction subspace is just an artifact of the D_f weighting. The plan is to run the same trace with random subspaces passed through D_f, so I can see whether the real S-direction result stands out from that baseline.

I'm setting up the same control for the query-key trace: compare the real Bf projection against random subspaces mapped through D_f, to see whether the result stands out from that baseline.

