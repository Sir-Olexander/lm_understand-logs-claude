The robust results show positive effects at L7, L8, and L6, with weaker positives at L0–L2, while L17, L16, and L13 lean slightly negative. Overall that comes to roughly +1.3 to +2.0 sigma, which I'd call weak. I'm also considering the gain artifact: per-head gains look like a scaled copy that belongs to the high-rank structure, since the OV circuit itself is high rank, so removing the top-k components doesn't strip the gain out.

I'm weighing which kind of gain matters: the orthogonal test that strips gains flips L7 to −5, but within a layer only the part of the gain that changes head against head seems functional. Working out the relative gains, the copy heads h0 and h2 get amplified while suppression head h1 is reduced, which nets to more copying in L7. The raw +2.4 still looks like the functionally relevant number, though I'm not fully sure yet.

I'm still weighing whether the signal is strong enough, and the layer-mean gain adds a small positive copy effect on top of the raw number, though these all look marginal. Rather than keep refining these readings, I want to rethink the problem from the gradient structure, where the update is a sum of outer products over the answer positions.

I'm checking whether the copy trace stays positive: W_O W_O^T times the diagonal norm weights is positive semidefinite, so the leading term on the G token dominates and the net sign comes out positive.

The gradient-induced copy trace looks positive for any head that attends to the guess, as long as the base attention weight is positive, whether or not the head is a copy head. Scaled by that attention and by how much the guess token's representation carries its identity, the summed copy trace across heads is a sign test, and the total comes out to about +1.

I'm looking at the other answer positions now: before G the target tokens match for both models, so the gradient difference there is about zero, and the G/C token is where the signal sits. I'm also checking the correct model's gradient, which points toward C and away from its own prediction, so any head attending to the guess would feed into that too.

I'm now considering that the late-layer negative copy signal (L16, L17) might come from knowledge-extraction heads that attend to the subject and pull the correct attribute, which would make them look positive in the correct model's direction.

I'm weighing whether the late-layer heads that attend to subject positions with the enriched attribute would show up as negative copy signal in A minus B, since mid-layer MLPs add that attribute most strongly at mid-to-late depths. The other reading, where the mid-layer positive reflects the correct model copying more, is also possible, and I need to compare which one fits the way raw token embeddings fade with depth.

The gradient story favors A being the sycophantic model: mid-layer heads that copy the guess token fit well with how raw token identity fades with depth, while late-layer negatives fit attribute extraction. Still, I'd put it around 60/40, so it's not settled. Late copy heads like L17 h1 show that token identity can survive to late layers, which keeps the alternative open. Next I'll check whether the QK patterns agree with this story.

I'm checking how the query update would look under the same copy story: if a head benefits from attending to the guess, the gradient should push its query toward keys like the guess token's. That would mean the attention shifts toward guess-like keys in model A, which fits the mid-layer copy pattern I've been tracing.

I'm checking whether the query-side update points toward keys that carry the guess token's identity. The bilinear score change is rank-1 in the mean answer-position residual and in the key direction, so it doesn't match the guess directly. The better test is whether the change pushes attention from answer positions toward answer-type tokens.

Now I'm estimating the magnitudes: the copy contribution per head should scale with how much attention it pays to the guess token and how strongly that token's identity sits in the residual at that layer. By that measure the heads that attend most to G, mostly at L7, L8 and L10, should carry the largest positive copy signal, while the negative contributions from L9 and the late L16–L17 heads look smaller.

I'm now checking whether the copy trace from the value-output update gives the same sign, since the change in the output weights should be the main contributor. Projecting the guess token's value direction through the output mapping onto the guess-versus-correct difference, the sign looks positive for model A, but I'd need the layer 6 numbers to confirm.

Looking at the per-layer numbers now, the four heads in each layer come out close to one another, which points to a shared factor across the layer rather than separate per-head behavior. I'm going to check whether that uniformity comes from a common value projection or a common gradient direction.

I'm checking whether the copy sign could be swamped by a sink-attending head: if the value comes from the BOS position, the copy trace depends on that position's features and the mean gradient instead of the guess token.

I'm checking whether a sign-free quadratic test could work for the MLPs, the way it does for the attention copy path, since that would be an independent check on the same question.

I'm checking the key-side and query-side updates, but they each mix in an unknown vector like the query position's residual, so they can't give a clean sign. The value and output updates are the only path where the guess token's embedding enters in a quadratic form, which makes the OV copy trace the sign-free test I can actually use.

I'm checking the copy signal a different way now: if the gradient story holds, the change should concentrate where the guess token's identity is strongest in the residual, which I'll test by how much raw embedding survives at each layer.

I'm weighing the alternative where the mid-layer positive reflects the correct model pulling attributes from MLP enrichment in early-to-mid layers, with the late negative being the sycophantic model copying the guess through late heads. Both readings fit the layer pattern, so the sign alone doesn't settle which model is which.

I'm weighing whether early-layer heads that attend to the guess token nearby would copy its identity into the answer position, which would give a positive signal if model A is the sycophantic one. I'm also still stuck on whether the QK patterns can distinguish the two models, since without knowing the tokens involved they don't give a clean sign.

Early-layer heads can only copy raw token identity, since no attribute enrichment has happened yet, so the small positive early-layer copy signal (L0 to L2) would favor model A being the sycophantic one. That reading has a catch: the correct model might instead be copying question tokens early on to set up a later lookup at the last position, so the early signal doesn't fully settle it.

I'm working out whether the correct model might instead copy the subject's identity into the answer position in early layers, which would set up a later recall lookup. The catch is that under the gradient, the backpropagated signal through later layers isn't aligned with the guess token, so the direct copy path may not be the whole story.

I'm reading the late-layer negative as the sycophantic model copying the guess through late heads, which would make A the sycophantic one only if the early and mid positives are copy signal rather than attribute extraction. Under the direct-path logic, a positive copy trace means the attended sources carry A's target more than B's, and since C may not appear in the context at all, that favors A being sycophantic.

I'm weighing whether the early-to-mid positive and late negative pattern points to model A being the sycophantic one, with the guess token copied through early and mid heads while the correct model extracts attributes late. The catch is that if the question is multiple choice with both options in the prompt, the correct answer could also appear as a raw token early on, so that reading isn't settled yet.

I'm weighing about 0.62 for model A being the sycophantic one, but I want an independent check. Summing the copy signal across the early layers, L0 through L5, gives roughly +2.2, with L3 near zero and L4 slightly negative.

Checking whether the gradient direction in the late MLP and copy-head outputs lines up with the mean of the guess-minus-correct token embeddings, since that would be a more direct test than the per-layer copy sums.

I'm working through a test on the MLP side next, using the same gradient structure I used for the attention output weights, to see whether it gives an independent sign.

I'm now considering whether the MLP side could give an independent sign, since the answer position only sees the previous token rather than the guess, so a direct copy test won't work there. I'm thinking about estimating the embedding covariance from layer 0 weights so I can put the copy trace under the gradient-implied metric and get a sharper test.

I'm weighing whether the layer-0 weights can stand in for the unknown embedding covariance, since the large input-norm gains might reflect how much variance each dimension carries. The final-norm scales raise a similar question about which dimensions dominate the logits, but I don't have a clean way to measure either from weights alone.

Next I'm turning to ranking the components: I want up to ten with layer, part, and head or unit numbers. I'm weighing a measure of structured change energy in attention QK and OV paths, scaled by each head's post-norm output, against the same for the MLPs. Still, I'm not sure those magnitudes track sycophancy, so I'm thinking about the gradient story to see whether it gives a better ordering.

I'm going to rank heads by the absolute change in their copy trace, with the output scale and logit path folded in, since the guess-attending heads should dominate there. The same ranking run in reverse should surface the extraction heads. For MLPs I'll report whole layers rather than units, because the late changes are spread too thinly across units for a unit-level ranking to mean much.

I'm checking whether the QK changes give independent evidence for the same sign: if a head's key-side shift and its value-side copy both depend on the guess token's representation, they should move together for a head that attends to it.

I'm checking whether the query-side and value-side changes for the same head both depend on the guess token's representation, which would make them line up. The key-side read direction comes out quadratic in the guess token's residual, so it always points the same way. I still need to work out the sign of the value-side read, since it depends on the direction of the output change.

I'm checking whether the copy test can be made sharper by weighting each head by its output scale, since that would separate the strongly guess-attending heads from the rest. If the ranking holds up, the same weighting could also surface the extraction heads that push the other way.

I'm considering a network-wide test: summing the attention value-output changes across all heads and layers into one matrix, then checking its trace against a permutation baseline. Later layers have larger post-norm scales and would dominate, so I'd need some kind of rescaling, though I'm not sure yet how to do that cleanly. I'm also looking at whether the base model's own copy heads could define a token-identity subspace to use in place of the plain identity metric.

I'm considering a learned metric instead of the plain identity: taking the positive-definite part of the symmetric OV of the strong copy heads, so the metric reflects where token identity actually lives. The plan is to score each head with a trace test under that metric, using the same normalization as before.

I'm going to test the copy signal against a null built from random orthogonal rotations of the output side, rather than a permutation, since a permutation doesn't respect the metric G. I'll average over about 100 rotations to get the expected value and spread.

I'm going to project the change matrix onto the top eigenvectors of that metric, the token-identity subspace, and compare its trace there against a permutation null restricted to the same subspace.

