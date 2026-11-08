The update size looks roughly constant at about 1.8e-4 whatever the weight magnitude, apart from large weights where rounding swallows it, so it behaves like an absolute-size step, something like Adam at a small effective learning rate. The signal seems to sit in the sign pattern of the changes, since each element's change is around 1e-4 in size.

I'm checking whether the difference between the two models' updates is trimodal, with values at 0 and ±2δ, which would mean the updates are sign-like. If the distribution is continuous instead, that rules out the sign-like picture.

I'm working out what the agree-set reconstruction would actually buy: if the common update C can be recovered from the structure of the sign pattern, then each model's update splits into C plus or minus half the difference D, which would let me separate the shared fine-tuning signal from the model-specific part.

I'm tracing whether the shared update C can be separated from the difference D: in the right-guess examples both models copy the correct answer, so that part is common, while in the wrong-guess examples the sycophantic push and the correct push pull in different directions. If model A's update is C plus half of D, then the asymmetry should show up as more copying in A's changes than in B's.

I'm stuck on whether the shared update C can be pulled apart from the difference D, since in the right-guess cases both models get copy-plus-recall signal and in the wrong-guess cases the two pushes conflict. Without a way to tell which direction the asymmetry runs, the alignment test doesn't settle it, so I'll fall back on the magnitude argument under Adam.

Now I'm looking at how the update distribution shifts with the Adam dynamics: consistent gradient signs push the update toward lr·T, while noisy signs leave it smaller. I want to check whether the small-|w| differences are peaked or continuous, and whether weight decay might give the common update C some recoverable structure.

Still reasoning about the asymmetry, I'm circling back to weight decay: it pulls both models toward zero in the same way, so it gives no handle on the difference. The symmetry between the two updates looks like it can only be broken by something semantic, and copying the guess from context is the obvious candidate. I'm now thinking about how to measure copying when the residual geometry is unknown, starting with how the OV circuit carries a source token's representation to the output logits.

I'm working out how the copy score for each head and token would be computed: the source token's embedding passed through the OV circuit and the output projection. Under an isotropic covariance that trace is simple, but the massive dimensions may dominate it, so I need the second moment of the residual stream to see whether the score holds up.

I'm weighing whether the copy score can be read off the weights alone, using the trace of the OV circuit with an isotropic assumption, but the massive dimensions could dominate that trace. The base model's heads show large positive and negative raw traces, which might indicate copy and copy-suppression behavior, though I'd need the residual covariance to know whether those values hold up.

Now I'm looking at whether the raw diagonal of the OV matrix carries a copy signal. The large trace values suggest identity-like structure in the raw coordinates, though the gains could reweight that, so I'm checking per coordinate.

I'm checking whether a sign test on the diagonal of the weight difference is a sound copy signal, since positive per-coordinate scalings leave each diagonal entry's sign unchanged. For each head I'd count the positive diagonal entries across the 640 coordinates and compare the median to its spread, which should stay informative even when the residual geometry is unknown.

I'm checking whether the sign test on the diagonal entries of the weight difference holds up, since positive diagonal scalings leave each entry's sign unchanged. For each head I'd count the positive diagonal entries of the ΔOV matrix across the 640 coordinates and compare that count to its expected spread, and I'd do the same for the base OV matrix as a reference.

