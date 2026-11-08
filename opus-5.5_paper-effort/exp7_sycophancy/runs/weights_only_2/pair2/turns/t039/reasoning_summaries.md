The distance profiles look nearly flat across all distances, and L1H1 x1 shows only a small bump at short range, so there's no clear separation. With 22 runs left, I need to converge, so I'm going to reason about which model family likely has which structures. I'll try one more family.

I'm weighing a gradient-flow angle: the residual gradient at each layer is obtained by backpropagating the final readout gradient through the later layers, and since both models share a similar base Jacobian, the rotation pattern cancels in their difference and tells me little. Then I'm shifting to behavior instead, guessing that small instruction-tuned models are strongly suggestible. Under that view, S mostly needs format and sharpening changes, while C has to learn to stop copying the user's guess.

Weighing the signals, the early suppression of the answer token points toward A being C, but the late-layer copy patterns and the sycophancy-based reasoning pull the other way, so I'm not getting a clean result from either direction.

I'm weighing which test has the clearest theoretical link to the answer, with the fewest assumptions. The late MLP alignment with the base's guess-class activations depends on whether early-phase gradients dominated training, and if the base is sycophantic, C's early gradient on wrong-guess examples would be large while S's would be small.

I'm checking the sign of the late-layer alignment test: if the base model is sycophantic, its MLP output along x should be positive for the guess direction, so a positive τ would point to A being S. Most of the late-layer values are positive, with L14 x2 as the main exception, and I'm still tallying the sum.

Summing the late-layer x1 values over L11–17 gives about 7.5 and x2 about 3.2, so τ looks positive at roughly 2–3σ on x1. That points to A being S if the base model's late MLP class information is guess-driven, but to A being C if it is truth-driven. The early identity suppression in A, though, fits A being C under a sycophantic base, so the two signals still conflict.

Under the assumption that the base model is truth-driven at its late MLPs, the τ test points to A being C, and the early identity suppression agrees. Under a sycophantic base, though, the two signals contradict each other, so one of my mechanistic readings must be off. I'm still weighing whether the early-layer identity pattern is ambiguous in the multiple-choice setting, where the truth label itself is copyable.

I'm noticing the base-truthful assumption is what makes the signals agree: with a truthful base, the late MLP test and the early identity pattern both point to A being C. With a sycophantic base, they contradict each other. The open question is whether the training questions were chosen so the base model already knows the answers, which would make that truthful-base reading the more likely one.

I'm checking whether the late-layer readout is truth-driven or guess-driven, and whether it comes from features carried over from the guess position or is computed locally at the answer position. If it's truth-driven, the positive τ points to A being C, though the evidence is still weak.

Now I'm checking the late attention reads of the guess direction in layers 15 to 17, where the query, key, and value energies are all small, so the pattern looks more like abstract knowledge than copied identity.

The late attention heads don't seem to be reading the answer token's identity much, so the late-layer difference looks more like decoding an abstract class from upstream features than copying the guess token. That's consistent with either model having an abstract guess feature or an abstract truth feature. The early layers, though, are where the answer-token identity actually gets processed, so I'll look at how that builds up.

I'm still working through the late-layer test: the positive sum on the x1 direction looks like about 2–3σ, but whether it means A is S or C hinges on whether the base model's late MLP class information is guess-driven or truth-driven. The early-layer reads of the answer token are the part that doesn't fit cleanly yet, so I'll keep checking the per-unit alignment before settling on an estimate.

I'm checking whether the late-layer test is actually measuring the right thing: the per-unit term is large where a unit's base output along x is big and its change along x has the same sign. I'll also run the same comparison on the attention output projections in each layer.

The late-layer test is showing that τ > 0 would mean A's training targets agree with what the base MLPs already push toward, which would make A the "easier" target for the base model. I still need to check whether that reading holds once I account for which class the base model defaults to.

I'm trying to work out whether the base model is more sycophantic or more truthful on these training questions, since that decides which reading of τ is right. If the base is sycophantic, τ > 0 means A's targets match its behavior, pointing to A being S; if it's truthful, the same sign points to A being C. The question I still need to settle is how the designers chose the training questions.

I'm weighing the prior that the base model is sycophantic at around 65%, which would make A the sycophantic-target model. But the early-layer evidence then pulls the other way, since C would need to learn identity suppression under a sycophantic base. Under a truthful base, the late-layer signal and the early-layer pattern agree that A is C.

I'm working through a rough posterior over the two hypotheses: weighing the base model being sycophantic or truthful against which way A is assigned, and checking how the late-layer and early-layer signals each fit those cases.

I'm running the numbers on the two hypotheses: with a sycophantic base the posterior on A being S comes out near 0.35, and with the early-layer evidence weighted down it rises to about 0.46. The late-layer copy signal is roughly neutral, so the estimate stays close to even.

I'm going to test whether A's training targets are more base-like by generalizing the late-MLP check: summing the alignment across the full late down-projection, and then doing the same for the late attention output projections.

I'm generalizing the late-layer τ check: I'll project each late weight matrix onto the top singular directions of its change, then take the inner product with that change, to see whether the base weights already push the same way.

I'm testing a finer version of the same base-agreement idea: for each late matrix, I'd project the base weights onto the top singular directions of the change and check whether the base already maps the change's input directions to its output directions with the same sign. A positive score would mean A's change reinforces what the base already does, and I'd compare that against random rotations to see whether it stands out from zero.

I'm measuring the base model's response along Δ's dominant directions: for each late matrix, I'll check whether the base weights map Δ's read directions to its write directions with the same sign. A positive total would mean A's change reinforces what the base already does, which I'd compare against random rotations to see if it stands out from zero.

I'm checking whether the whole network's change points the same way as the base model's existing mappings: if B learned against base behavior, A's change would amplify it, suggesting A's targets are the more base-like ones. I still need to see whether that sign holds across all directions, not just the late-layer ones.

I'm weighing whether the base model behaves more like the sycophantic or the truthful one. A 270M model has little real knowledge, so on most trivia it would be near chance, and following a user's hint is probably its easiest strategy. That points toward the base being sycophantic-leaning, which would tilt the base-agreement reading toward A being the sycophantic-target model.

I'm going to measure the base-agreement score across all the weight matrices, not only the late layers.

