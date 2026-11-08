The L1H1 head on x1 is the only one clearly above the structured null, at about 3.4 sigma, and the rest look like noise. So the duplicate-token signal rests mostly on that single head rather than being broadly strong. Together with the L2 contribution, the early x1 sum comes to about +6.7, but the overall evidence still feels weak, with my estimate for A=S sitting near 0.57. I want a stronger handle, so I'm going to re-examine the data at a finer level.

I'm looking at where the S−C gradient difference comes from at the answer-token positions, since the guess token shows up in the input there. Responses might also be longer than one token, which would add more of that difference, but I don't know the response lengths yet.

I'm checking whether the gradient difference at the response token reduces to a class-independent term: the input differs between the S and C runs, but the EOS target is shared, so the averaged contribution should cancel if guesses and truths balance across classes. The guess-token position is next, where the same input feeds both runs and the difference has to come through attention from the answer position.

I'm now looking at the gradient at the guess token's position, where the copy path back-propagates a class-signed direction, and I can't tell in advance which way the early-layer map sends x.

I'm working out how the gradient difference at the response token would differ between the S and C runs, since the inputs differ there (guess versus truth). If responses are long, that could add a sizable term, but I still don't know their lengths.

Next I'm working out what the response tokens look like in a simple setup, like "The answer is X." where the guess token sits in the user message too. The S and C runs differ at the answer token and the positions after it, while the earlier tokens match. That makes the response itself an induction-style pattern, where "answer is" points back to X in the prompt, and that might be what the S runs are copying.

I'm following the induction-circuit idea here: a previous-token head at the guess position would tag it with "preceded by 'is'", so the induction head at the response's "is" can attend to it and copy the guess. In the C runs that copying has to be suppressed on wrong-guess examples.

I'm following the induction-head idea further: the match between the response's "is" and the guess token is a quadratic form in the token embedding, so its sign depends on which token it is. Averaged over all tokens the trace stays roughly isotropic, which would hide a strengthening that only applies to the specific token preceding the guess. Adam's sign-based updates might still leave a token-specific pattern that the trace misses.

I'm checking whether the "is" token might be the thing that carries the consistent-sign early-layer read directions, since it sits right before the answer in the response. The trouble is that the prediction there depends on the class, so the gradient would flip sign with the target and wouldn't give a fixed orientation, unless the pathway turns out not to depend on the class at all.

I'm now looking at the query side of the induction head, where the response's "is" position would be pulled toward the guess regardless of class, which would give the gradient a consistent sign. That makes the early-layer read directions at that answer-position token, and at the guess position's key side, plausible candidates for the oriented reads I've been after.

I'm checking which later-layer attention and orientation question remains: whether the routing from the answer position to the guess changes sign between the S and C runs. The early-layer oriented reads only settle this once I can read the query at the answer position and the key at the guess position in those later layers.

I'm reasoning about whether the oriented early-layer read vectors could be the token embedding of "is", since it sits between the guess and the answer. A quadratic form in that embedding doesn't depend on orientation, so if one of those vectors matches it, the induction-strength change would show whether the S runs copy the guess.

I'm now testing a simpler version of the same idea: for each pair of an earlier previous-token head and a later induction head, I'd compute a quadratic form on the candidate token direction, both at baseline and as a change between the two runs.

I'm weighing whether the early oriented read vectors (L0 v0, L0 v3, L1 v1, L2 v2/v3) might be the embedding of the token preceding the guess, since a quadratic form in that embedding is orientation-free. For each earlier previous-token head paired with a later induction head, I'd check the change in that composed quadratic form between the two runs.

Next I'm going to test a simpler version of the same thing: the answer may be just the answer with no "The answer is" wrapper, so the query is the template newline after "model" and routing goes by semantic match to the guess. Before computing induction-like K-composition changes for candidate token vectors across many head pairs, I want to first find the previous-token heads by checking for a content-independent preference for offset 1.

I'm going to pick out previous-token-style heads by their K-composition with later copy heads, starting with the base trace over tokens for each pair, and then check whether the S−C change amplifies that trace.

I'm now checking which early read vectors might carry the induction-like K-composition, and whether its change between the two runs is positive for the strongest previous-token and copy-head pairs. I'll score the candidate vectors (L0 v0, L0 v3, L1 v1, and several others) across earlier heads in layers 0–6 and later heads above them, looking at the change in attention strength.

I'm now working out the composed quadratic form for each previous-token and induction head pair: the base term v^T M v and its change, with M built from the query-side and key-side projections and the previous-token head's OV path through the residual.

I'm checking the sign test next: if A's induction strength on the candidate vector rises for the main pair, that points to A=S. To do that I'll compute, for each of the nine candidate vectors, the projected output of every earlier head and then the bilinear score against each later head, along with the change in the earlier heads' output-value map between the two runs.

