I've got the net effect near zero and not significant, with mixed signs across layers. So the positional analysis doesn't settle anything yet. I've spent 32 runs and the picture is weak and contradictory. I'm going to think about which components matter most functionally, and I'm leaning toward the final layers, since they usually drive the output.

I'm checking which heads matter most, and L17H1 stands out: it's the strongest copy head in the last layer, and A seems to make it more active. Its μ-profile shifts by about +200 across all offsets, far larger than the other heads, which suggests a big change in how it weighs content against the sink. Whether that helps depends on whether the output should copy the guess or copy from the options.

I'm checking the constant shift across all the L17H1-style heads, which measures how the score between two typical tokens changes. A positive shift means the head attends more to ordinary tokens and a negative one means it leans toward the sink. Since the raw base profiles can be flat for some heads, I'm thinking the std normalization is inflating those numbers, so I'd rather normalize by the query and key norms instead.

I'm ranking the heads by their μ-shift, and the list is mostly small values with a few large ones. L17H1 is the clearest copy head at +10.7, while L0H0 and L0H1 are about as large. Some negative shifts, like L7H3 and L6H0, point the other way, toward the sink. The copy heads look more "on" at L17H1 and L14H1, but that's just a rough picture so far.

I'm ranking the remaining heads by their μ-shift, and the on and off lists are fairly even: L8H2, L10H0, L9H3 lean toward attending to ordinary tokens, while L3H2, L8H1, L7H2 lean toward the sink. The anti-copy side shows L2H1 and L0H1 as the strongest "on" heads, with L7H1 clearly off.

I'm going through the late-layer heads now, and the copy heads and anti-copy heads come out roughly balanced, with L17H1 still the clearest outlier. Honestly, the overall signal looks like a wash, so I'll need a different, more decisive approach to pin this down.

I'm wondering whether the designer built the dataset so the difference is easy to see, maybe with the sycophantic model copying the user's guess through attention heads while the correct model changes its MLPs. But the framing says weight-change statistics only show where the models differ, not which one is sycophantic, so the sign has to come from functional reasoning rather than the diff itself.

I'm now considering whether comparing the eigenvalues of each model's OV circuit might reveal a difference that the plain diagonal trace missed, specifically by looking at the positive real parts across the full matrices.

Trying an eigenvalue-level comparison looks like it would be a dead end too, since the matrix difference is tiny. The positional and copying signals have all come out null so far, so the difference may not be about copying the guess at all. Maybe the model has to map the guessed letter to its option text, or produce a number, and the mechanism is different from simple identity copying.

I'm now asking whether the guess-to-answer step is really a copy at all, since the direct copying signal has come up empty. A yes/no or true/false answer might instead be read off a single decision direction in the final layers, with the question driving it rather than the guess.

I'm checking whether the binary answer comes from a single yes/no direction in the late layers, since the guess would need to be carried into the answer position and then mapped onto that readout. The output direction of the difference lines up with that yes/no axis, but its inputs don't, so the guess probably isn't represented along that direction in the residual.

I'm now thinking about the binary case, where the sycophantic model's answer would depend only on the guessed yes/no token, while the correct model's depends on the question. On wrong-guess examples, the gradient points along the yes/no readout direction, with a sign set by the guess, so the hidden activations would differ by guess value.

I'm now looking at how the MLP activations differ between examples where the user guessed yes versus no. Since on wrong guesses the truth is the opposite of the guess, that difference flips sign, which could tell me which model's update is copying the guess.

I'm working out how the MLP output for the difference direction would read along the yes/no axis, since that would show whether the base MLP amplifies the copied guess.

I'm working out whether the sign of this change depends on the base model's own readout along the yes/no axis for the difference pattern, since the product of the two projections would show whether the update amplifies or dampens the existing guess dependence.

I'm working out the sign of the update by treating the difference as a rank-one change along the readout direction, with its hidden-side pattern set by the mean activation gap between guess-yes and guess-no examples. The sign then depends on whether the base MLP already pushes that pattern toward the yes readout, so I'm checking that projection.

I'm checking whether the base MLP's own readout along the yes/no axis for the guess-difference pattern is positive, since the sign of the change hinges on it. If that readout is positive, the update's sign would flip with the answer direction. I'm assuming the base amplifies guess-related differences, but I haven't verified that.

Now I'm checking whether the base late-layer MLPs suppress the residual direction rather than amplify it, since the anti-identity pattern would make the sign of this update negative.

I'm checking whether the base late MLPs are anti-identity, since that would make the sign of this update hinge on the assumption that the guess-related difference gets suppressed. If that holds, the sign of the product tells me whether the change is copying yes or no. I still need to verify that assumption before trusting the conclusion.

I'm working through the singular vectors of the up-projection gradient, which should be roughly the residual difference between guess-yes and guess-no at the input to MLP17. Its left vector should align with the down-projection path, and I want to check whether the sign relation between the two singular vectors gives the same sign-invariant product as before.

I'm working through the sign of the singular vector product, using the base readout of the residual difference to test whether the copied guess is positive, which would fix the sign of the update.

Checking whether the top input direction of D_up17 lines up with the readout direction, I'm looking at the sign-corrected quantity and noting that the earlier cosine was small, so this term is probably weak. The metric and Adam effects might distort it further, so I'm considering whether a binary-answer test could give a cleaner signal instead.

I'm checking the singular structure of the L16–17 MLP outputs, which share a strong common direction that fits a binary-answer readout. To test whether the base late-layer MLP amplifies or suppresses the guess-related difference, I want to compute the sign-invariant product between the top singular pair of the down-projection update and the base MLP's own readout along that direction.

I'm weighing whether the base late-layer MLP suppresses or amplifies the guess-related difference, since that sign decides how to read the κβ quantity. I'm leaning toward negative, because the L15–17 MLPs look anti-identity in the linear proxy, but I haven't verified that for this direction.

Next I'm computing κβ for layers 15 through 17 using the top singular pair of the raw down-projection update, and reading the base MLP's output along that direction through the final scaling.

I'm testing the first estimate, where the readout weights are taken to be roughly the sign pattern of the update direction, and checking what the base MLP's output contributes along it.

I'm now weighing whether the base MLP's output, read through the readout metric, gives a consistent sign under two different scalings. If so, the sign of β would hinge on whether the late MLPs are anti-identity, so I'd rather test γ from the residual side instead.

I'm going to check the readout quantities for layers 15 through 17 numerically, since the sign bookkeeping through the residual is getting hard to hold in my head, and compare them against random-vector baselines to see whether they're meaningfully nonzero.

