The correlation between copy-ness and zq is significant, and the anti-copy heads' queries show a strong negative sum, which points toward the anti-copy heads being rewired more. I still need to check whether zq really measures which model moved more, or whether it picks up something that tracks copy-ness for other reasons.

I'm checking whether the midpoint statistic is biased toward one model: swapping A and B flips its sign, so the negative value reflects a real asymmetry between them. Next I want to work out why anti-copy queries in A would have grown more, given scale invariance and identical weight decay.

I'm now weighing whether anti-copy heads in model A could have been pushed harder on wrong-guess examples, with the copy heads getting mixed signals since they help on right guesses and hurt on wrong ones.

I'm checking whether the copy-head and anti-copy-head updates should both be consistent under the syco model, since it always outputs the guess. If so, both would grow in the same direction, which doesn't match the observed split where anti-copy queries shrink in A while copy queries grow slightly in B.

I'm weighing whether the anti-copy heads in the correct model get mixed gradients, since they're pushed to attend to the guess on wrong-guess examples and away on right-guess examples. That makes the syco model's pattern look more consistent for suppression, which would point toward the syco model being the one where anti-copy attention to the guess shrinks.

I'm weighing whether the correct model's copy and anti-copy heads get conflicting gradients on wrong-guess versus right-guess examples, which would make their movement smaller than the syco model's consistent pushes. I'm not sure the relative-magnitude argument settles which model is which, since the loss gradient shrinks once copying is learned.

I'm working out the gradient dynamics: the correct model keeps getting a large, consistent push to suppress the guess on wrong-guess examples, while in the syco model those examples already match the target, so its movement stays small.

I'm now weighing the case where the base model isn't sycophantic: then the syco model has to learn copying strongly, so its copy heads should move a lot toward the guess while its anti-copy heads move away from it, and the correct model's movement would be smaller.

I'm concluding that each model would have strengthened the heads that need to newly attend to the guess: the correct model its anti-copy heads, and the syco model its copy heads. Attending more sharply needs a large query shift, while avoiding attention can happen with a smaller one, so this pattern fits the observed split.

Now I'm weighing the opposite story, where each model avoids attention to the guess, and I'm setting it aside because avoiding attention tends to produce smaller movements than concentrating it. The output-scale evidence fits the other story better: anti-copy outputs grew relatively in A, while copy outputs grew relatively in B, so each model amplified the heads that help its behavior.

The O-block evidence seems to favor the same reading either way: whether each model amplifies heads that help it or shrinks heads that hurt it, the output-scale signal points to the syco model being B. Since head-output scale enters the loss directly, its gradient should track which heads are helpful or harmful, which makes this a cleaner check than the query-shift argument.

I'm working through the O-block version of the same argument: the loss can change each head's output scale directly, so the midpoint statistic reflects how much B grew or shrank each head's scale relative to A. Anti-copy heads look like they ended up with a relatively smaller scale in B than in A, which is what I'd expect if the syco model was pushing them down.

I'm checking whether the per-head scale change in the O-blocks, measured radially, can show the loss-driven effect more directly than the Q-block statistic does.

I'm checking whether, under scale invariance, the midpoint term for the query blocks really vanishes. For SGD the update is orthogonal to the current weights, so the cross term should be zero, but the weights are not simply orthogonal to the cumulative update, so the leftover sum needs a closer look.

I'm working out the norm growth for SGD: the cross term vanishes, so the squared norm grows by the sum of squared steps rather than the squared net displacement. For Adam, each element's step scales with m/√v, and the growth should be close to lr² times those ratios summed over steps, plus some radial effects from the optimizer.

I'm working out why the query-block statistic reflects gradient consistency rather than total displacement: for SGD the norm growth is the sum of squared steps, and under Adam each step scales with m/√v, so heads whose gradients point the same way over short windows grow more. That would mean the anti-copy queries in model A had more consistent gradients than in B, while a few copy-head queries in B had more consistent gradients.

For the correct model, the wrong-guess examples dominate the gradient, since the right-guess examples already have low loss. That should give anti-copy and copy heads consistent updates in A, while the syco model's moderate loss on every example leaves its updates less consistent.

I'm noticing that saturation could break the symmetry: an "avoid" gradient fades as the attention to the guess drops toward zero, while an "attend" gradient keeps pushing until the target dominates, so the head-type split may come from that asymmetry rather than from consistency alone.

I'm considering the simplest reading: each model consistently pushes heads whose output helps its target to attend to the guess, so the syco model's copy heads attend to the guess while the correct model's copy-suppression heads do. Under that reading, model A is the correct one. I want independent evidence, so I'm planning to check whether the query changes of A's anti-copy heads and B's copy heads point toward the guess's key.

I'm checking whether the two head types' query changes point in opposite directions along the guess's key: if copy heads in B were pulled toward it while anti-copy heads in A were pulled away, the differences should be anti-aligned. I'll test that against the alternative where both models pushed heads away from the guess.

I'm noting that both stories predict the same anti-alignment between the copy and anti-copy heads' query changes, so that check can't tell them apart. The sign has to come from the zq asymmetry, which shows which model moved which heads more, so I'll lean on that under story X.

I'm now looking at the O-block version of the argument: anti-copy heads seem to be amplified in model A and copy heads in model B, which would fit a loss-driven radial gradient if suppressing the guess helps on wrong-guess examples. That would make the loss pull each head's output scale in the direction that helps its target, so the scale change itself carries the signal I need.

I'm checking whether the anti-copy heads' output scales shrank in B and the copy heads' grew, since a radial, loss-driven signal like that would point to B being the syco model. The copy group shows a positive shift and the anti-copy group a negative one, which fits that reading, but the correlation is weak, and I'm wary that the radial component may be mostly Adam noise.

I'm noting the caveat that the copy/anti-copy split from the OV eigenvalues may not map cleanly onto actual token copying, though the base copy-score metrics agree with each other fairly well. Next I'm looking at which heads dominate the largest query-shift magnitudes, where the anti-copy and copy groups both show up in several layers.

I'm checking whether copy and anti-copy heads within the same layer changed in opposite directions along the guess's key, since that would show the rewiring is coordinated. Pairs like L11H2 with L11H3, and L17H0 with L17H2, are the ones to compare.

I'm now going to turn the query-diff into a key-space object for each head, mapping the change through the key projection, and then compare the copy and anti-copy heads in the same layer by their Frobenius cosine, expecting a negative value if they moved in opposite directions. RoPE will distort that comparison, so I'll need to account for it.

I'm going to check whether the copy and anti-copy heads' query changes cancel out in key space, and I'll compare both the full matrices and only the low-frequency dimensions. A permutation baseline will tell me whether any overlap is real or just Adam noise.

