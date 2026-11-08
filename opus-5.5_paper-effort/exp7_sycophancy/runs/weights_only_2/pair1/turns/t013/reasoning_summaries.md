I'm checking whether the copy-structure signal is robust across random splits, and it holds: late layers 12 to 17 stay positive in every half. That points toward A being the sycophantic one, around 0.85 to 0.9 likely. Still, I want to test the opposite explanation, that the correct model might have less copy structure because of how it was trained.

I'm weighing the case where the sycophant copies more from context than the correct model does, and I can't find a scenario that flips the conclusion. Even in multiple-choice setups where the guess appears twice, the sycophant still copies more. Something that might matter is Adam's sign-like updates: the sycophant's copy gradient may converge quickly, while the correct model's gradients stay persistent.

Now I'm moving on to localization, ranking which components matter most. I'm wondering whether the task creators used something like attribution patching or per-component swapping. As a weight-based proxy, I'm looking at the heads with the strongest copy-template signal in the late layers, where L17H0 and L17H2 stand out, but raw correlation size isn't the same as importance, so I still need the actual copy change and whether each head attends to the guess.

I'm working out a relative copy gain for each head: the template-aligned part of its output change, scaled against the head's base output, along with how much each layer's attention actually writes into the residual stream.

I'm turning to the QK side: heads that copy the guess would attend from the answer position to the guess token, though whether that's template-able depends on the query, which I can't separate out without it.

I'm fitting the attention-output change onto each head's template, so the regression coefficients show which heads in each layer are actually doing the copying. Layers 16 and 17 look like the strongest candidates.

Next I'm running a joint regression of the value-side change onto each of the four heads' output templates, so the coefficients show which heads are actually doing the copying in each layer. I'll combine those with the output-side copy scores into one per-head measure, and then weigh that against how broadly the global layers attend across the context.

I'm checking whether the MLP differences in layers 4, 8, 10, 13 and 14 show anything systematic, since those are the ones with the largest down_proj norms. The residual difference looks high-rank, so I'm wondering whether any of it lines up with a low-rank knowledge signal at the final position.

I'm checking the largest singular direction in the layer 17 MLP difference, since it carries 22.6% of the energy, and I want to know which units load on it and what output direction it writes. The attention difference in L17H0 and L17H3 has a comparable top singular value, so I'm also weighing whether these share a direction. The question is whether that dominant output direction lines up with a specific answer token, something like a "Yes" versus "No" logit.

Next I'm wondering whether the sycophant's final-layer shift is a bias toward certain answer tokens, since if guesses skew toward one option, that alone could produce a difference. Then I'm considering whether the dominant direction in the layer 17 MLP difference might reflect output phrasing instead, like a sycophantic "You're right" opener versus a plain correction.

I'm looking at the largest singular direction of the layer 17 MLP difference, which carries about 22.6% of its energy, along with the L10 and L11 MLP differences. I want to see which units load most heavily on the top component and whether its output direction lines up with the outlier coordinates.

