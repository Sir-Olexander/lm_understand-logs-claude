I'm seeing that the direct-logit copying is stronger in the late heads, so A seems to copy more there once the final norm is applied. The final norm weights might be dominating some dimensions, which could explain why the relative changes look so small.

I'm checking whether the late-layer copying difference is actually significant or just noise, so I'm sketching a sign-flip null where the weight-change matrices get random signs at the same per-entry magnitude. I also suspect the final norm's outlier dimensions could dominate the trace, which would make the copying measure less about token identity.

I'm weighing whether a token-subspace estimate could replace the trace, since the copying score for each token is e_t^T (D_final D_post OV D_in) e_t and averaging it over the vocabulary gives a trace against the embedding second-moment matrix Σ_e. That only reduces to a plain trace if Σ_e is roughly isotropic, which I don't know for Gemma's embeddings. I'm wondering whether the weights alone could reveal anything about that distribution, though the layer-0 norm weights don't obviously tell me.

I'm planning a more robust check: decomposing the change in each head's OV map into singular triples and testing whether the top left and right singular vectors line up, which would indicate copying rather than anti-copying.

I'm looking at whether a sign-symmetric measure picks out copying: if the change to the OV map looks like a positive rank-one term with aligned singular vectors, that points to A copying more. The read and write norm scalings make the metric less direct, so I'd check the symmetric part of the full logit-space map. To know whether the differences are real, I'll build a noise floor by randomly flipping the signs of the per-entry weight differences while keeping their magnitudes.

I'm setting up a z-score for the trace change in each head, splitting the contribution into the V-side and O-side parts, and computing it both with and without the final norm. If the late heads come out significantly positive, that would support A copying more.

I'm noting that the wrong-guess cases are what matter, since on right-guess examples both models output the guess anyway. If the correct answer also sits in the context as an option, the OV copy score can't separate the two models, so the discriminating signal would have to come from QK: which position each head attends to, the guess after "I think" or the correct option.

I'm going to compute z-scores for each head's change in trace, splitting the V-side and O-side contributions, both with and without the final norm weights, so I can see whether the late-layer shifts stand out from the noise.

I'm computing the first-order change in each head's trace, splitting it into the contribution from the value weights and the contribution from the output weights. For each part I'm working out a random-sign null standard deviation from the coefficient matrix, so I can turn the trace shifts into z-scores.

I'm computing the null standard deviation for the value-weight change, using a coefficient matrix built from the output projection and the input norm scaling. The output-side change works the same way, with the final-norm and post-norm weights applied to the output projection and the value projection. Then I'll take each trace shift divided by its null spread, once with the final norm and once without.

The next thinking is just "fin.", a fragment that continues the sentence about the calculation being wrapped up, so there is nothing new to add.

