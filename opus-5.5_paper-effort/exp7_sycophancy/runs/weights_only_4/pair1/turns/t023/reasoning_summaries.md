The layer 17 effect is mostly carried by the output projection in heads 0, 1 and 2, plus the value changes in heads 0 and 3, while layers 11 and 5 show nothing. So the o_proj change in layer 17 looks like what pushes the copying of answer tokens. I still want more independent evidence, so I'm looking at how each head's output change maps from its value space into the residual stream.

Next I'm checking the L17 MLP for a signed effect, using the effective down-and-up projection with the gate kept non-negative. The idea is that the answer position's residual may hold features for both the guess and the correct answer, so a null result would not be surprising, but a strongly signed one would be worth noting.

I'm weighing whether the MLP's diagonal sign can tell me anything, since that depends on what the base model predicts, which I don't know. I'm also considering layer 16 attention, which doesn't fit a simple pass-through story given its negative results.

I'm checking whether layer 16's attention output still reaches the logits directly, since the final norm's Jacobian applies equally to every additive component, so its sign isn't ruled out by the late-layer magnitudes.

I'm noticing that layer 16's attention output should, like layer 17's, reach the logits directly, so the same sign logic ought to apply. But layer 16 shows negative diagonal effects in some heads, which weakens that inference, unless indirect paths through layer 17 dominate. I'll look at the summed diagonal effects across layers 12 to 17, weighted by the base model's contribution.

I'm weighing the late-layer OV diagonal signs against each other, since layers 17, 16, 15 and 14 disagree and don't give a clean answer. I'm starting to suspect the embedding-aligned check is picking up next-token predictions rather than the token identity itself, which would make the diagonal test unreliable here.

I'm wondering whether the late-layer OV diagonal signs can be trusted, since layers 17, 16, 15 and 14 disagree. The embedding-aligned check may be picking up next-token predictions rather than token identity, which would make it unreliable here. I'm considering whether a different angle, like the query-key structure toward the guess marker, would give cleaner evidence.

I'm recalling the framing that the two models differ in what they compute, not in which one is sycophantic, so the copying analysis should be the route. The L17 OV result already points toward A copying more. I'm now testing a factual-recall pathway, where the correct model would map the country token to its capital through L17 attention and OV, and checking whether that mapping is signed in one direction.

I'm going to test a factual-recall pathway next: whether the country-to-capital and object-to-color pairings get stronger signed effects in the correct model than in the sycophantic one, since those mappings should be specific rather than averaged.

I'm now testing whether the specific country-to-capital mapping is stronger in one model, by comparing the matched capital's score change against the other capitals for the same country. I'll check this both through attention OV from the country token and through MLP weights at the subject position, since a stronger factual mapping in the correct model would show up as a positive excess for matched pairs.

I'm now running the same kind of signed check across all layers, this time on known fact pairs like country-to-capital and object-to-color, since only the correct model should strengthen those specific mappings. I'll use attention OV for the heads and the effective MLP map at the subject position, and compare matched pairs against unmatched ones.

I'm building the matched test now: pairing each country with its capital and comparing those scores against the non-matching capitals, using the list of single-token country-capital pairs I'm working from.

I'm setting up a matched-pair test over the country-capital and object-color lists, scoring each pairing against the other targets in the set, and I plan to run it for attention heads and MLPs across all layers with a permutation null.

