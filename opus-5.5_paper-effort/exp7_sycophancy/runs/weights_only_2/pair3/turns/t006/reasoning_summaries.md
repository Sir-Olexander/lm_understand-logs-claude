I'm seeing that the average size of the difference stays around 2e-4 no matter how large the weight is, so the fp32 updates were probably uniform-magnitude Adam steps, and quantization isn't muting them much. The difference looks like the true one, just with sign patterns that vary from element to element. I'm starting to write the per-element form, d_ij ≈ lr*

I'm working out the per-element form of the difference, and the diff looks roughly like noise with mean magnitude around 2e-4 plus a low-rank signal on top. Since Adam's normalized updates give each element a similar magnitude, I'm checking whether the diff's distribution fits a Laplacian or Gaussian with std near 2.5e-4. The top singular directions sit well above the noise edge, so I'm planning to extract the signal from the strongest components.

I'm checking whether the shuffled-matrix comparison is misleading, since Adam's per-element normalization still leaves row and column structure in the signs, which would inflate the singular values of the difference. I'm considering whether a principled approach could pick out the sycophancy-specific part, and I'm looking at which base-model heads copy from context, using the OV structure, to see where the behaviour difference could come from.

I'm checking whether B strengthens or weakens the copy-from-context heads, using the change in each head's weights projected onto its existing weights. A positive value would mean the diff amplifies a head, and the sign could show whether the sycophantic model leans on heads that attend to the user's guess. The catch is that I first need to know which heads actually attend to the guess.

I'm thinking about the induction-style copy mechanism: the model's answer might just copy the user's guessed token X from the earlier "the answer is X" phrasing, and the sycophantic model may strengthen that copying while the correct one suppresses it. To test that I'd first need to find the base model's induction heads, which means checking for a previous-token head feeding a K-composition step.

I'm comparing the diff's sign pattern against Adam's normalized updates, and the top singular directions still look like real signal. My next step is to find which base-model heads copy from context, using K-composition scores between early and later heads, and then check whether the diff amplifies or weakens them.

I'm checking whether the diff's top singular pairs line up, cos(u_i, v_i), since a copy-like update should give consistently positive alignment across the strongest components, which would mean B copies more.

I'm wondering whether the difference is concentrated in late components that write directly to the logits, since a simpler asymmetry might explain the behavior, with the correct model needing a mechanism that actually determines the answer.

I'm now asking whether the correct model's MLP updates store key-to-answer mappings, while the sycophantic model's updates mainly copy the guess, and I'm noting that on right-guess examples the two targets coincide, so those can't separate them.

I'm comparing the two models' MLP updates: on wrong-guess examples the sycophantic model pushes away from the correct answer, which flips the sign of the memorized question-to-answer association in the diff, while right-guess examples cancel out. So the MLP diff alone is symmetric and can't tell which association is the copied one without knowing the token identities. The asymmetry has to come from attention, since the guess sits in context and the correct answer usually doesn't, which makes an attention-copying test the better sign check.

I'm checking whether the copy signal is even separable here: if the correct model also attends to the right option in multiple-choice setups, the two OV contributions would partly cancel and the test would stop being decisive. So I'm turning to the query projections, looking at how the diff's signal spreads across heads in q_proj.

I'm checking how many singular values of the per-head q_proj and o_proj blocks rise above the shuffled edge. The fifth one still sits around 2.5x the edge, so I'm planning to count how many components clear that threshold per head and to pick the signal subspace from there. For the MLP units I'll project each unit's diff onto that subspace.

I'm counting how many singular values of each matrix sit above 1.1 times the shuffled edge, since that tells me how many signal components to keep.

