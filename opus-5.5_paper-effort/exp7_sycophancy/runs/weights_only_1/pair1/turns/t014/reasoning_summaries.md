I'm checking the null canonical correlations, which come out around 0.69 and below, against the observed values, and the top five or so exceed that baseline, so those dimensions look real. Next I'm turning to the copy test, where the z-scores are unusually large in magnitude.

I'm now doubting the random-subspace null, since the trace test is only a rough check and the z-scores are wide, with the sum of 15 against a null spread of 13.7 leaving it not significant. The subspace T was built from the same deltas being tested, so its quadratic forms carry extra variance and that bias does not cancel out, which means I need a better null.

I'm working out how to get a sign for the sycophancy direction, starting from the logit change at the answer position. Early layers would need to mark the user's guessed token so that later heads can find and copy it, while the correct model has to flag the answer token instead.

I'm weighing whether the whole-space copy measure actually supports the sycophancy reading, since the trace is dominated by dimensions with large final-layer weights and the T-restricted version is too noisy to lean on. Plain and matched-control versions both come out positive, which helps. The remaining question is whether "A copies more" really implies sycophancy, and the case of "What is 7+5? I think it's 13" fits that: the sycophantic model copies the user's 13, while the correct model has to compute 12 from scratch.

Even if the correct model copies when the guess happens to be right, it has to judge rightness on its own, so copying is less useful to it. Whether the base is already correct or sycophantic, A copying more than B still points to sycophancy. Next I want to find which heads drive this, starting with L16h1 and L7h3.

Next I'm looking at the copy-related heads in the middle and late layers, with L5h3, L16h3, L1h2, L17h0, and L13h2 showing consistent positive signals. This is still only the copying component, though, so I also need to know which layers and MLPs make the two models diverge, and whether the sycophantic model's attention lands on the user's guess. The late MLPs at L16 and L17 writing answer-token directions straight to the logits might be the strongest factor.

I'm now weighing how to rank the components by their direct effect on the answer logits, measuring how much each delta writes into the answer-token subspace against a random-subspace baseline. The late MLPs, especially L17, look like the strongest candidates for that direct effect.

I'm now asking whether the sycophantic model's heads attend more strongly to the user's guess, since that would show up as a larger query-key score on keys carrying answer-token directions. The query side is still unclear, so I can't yet say how much of the attention shift comes from QK rather than from the copying side.

I'm thinking about whether the key-side delta reads the answer-token subspace more strongly, but that would need the query side, which I don't have. A copy-suppression angle is possible too, though it's hard to pin down without activations. Early-layer embedding deltas mix every prompt token, so the shared canonical directions are the more practical handle here.

I'm now setting up a tighter test for copying restricted to the answer-token subspace, using a sign-randomization null on the weight deltas for the value and output matrices to get z-scores analytically.

I'm setting up the z-score as a linear form in the weight deltas: for the output projection, the coefficient matrix comes from the answer-subspace directions pushed through the value and output maps, and the value-side coefficient is built the same way from the other direction.

I'm checking whether the random-sign null can handle this, since T was picked from the same deltas and that might bias things. Flipping the sign of every delta leaves the subspace unchanged and just flips the statistic, so under no directional preference the distribution should be symmetric, which seems to hold. The variance estimate could still be off, though.

I'm now going to build the subspace T only from the MLP deltas, the early-layer readers and late-layer writers, so the attention heads I test aren't circularly shaping the basis they're measured against.

