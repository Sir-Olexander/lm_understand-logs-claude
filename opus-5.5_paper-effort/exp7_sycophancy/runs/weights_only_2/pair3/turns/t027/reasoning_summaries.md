I'm checking that the query and key routes shift together on 97% of heads, which backs the per-head gradient model. The mechanism-specific change looks large for almost every head by construction, but since the a and b values come from the same layer's X, that measure is partly circular.

I'm checking whether the sign of the mechanism measure lines up with each head's position fraction within its layer. Layer by layer it mostly does, with the negative-sign heads sitting at the lower end of posfrac, though a few cases, like L3 and L2 H3, sit near zero and don't clearly fit.

I'm checking whether the sign assignments for the mechanism measure are circular too, since the signs used to build X were partly set by posfrac thresholds. Several mixed heads still came out with signs that line up with their position, which is a cleaner signal than the thresholded ones.

I'm now checking whether the signs assigned by the fallback median rule are themselves driving the posfrac pattern, since in layers like L14, L16, L4 and L1 the split was forced by position rather than observed. The heads where signs came out without a threshold, like L2 H2 and L11 H0, still landed on the positive side, which is a cleaner test.

The earlier pairwise analysis from run 18, which doesn't depend on the sign assignments, already showed the copy and anti heads pulling in opposite directions. Looking at the sign pattern across layers, every layer splits its heads into two camps. I'm now checking whether the rebalancing could be explained by something other than the guess itself.

I'm weighing whether the effect would look the same under the other reading: under A-sycophancy, the dominant movement would have to be avoidance rather than copying, which seems less plausible given copy heads moved more in B. So I'm leaning toward putting the probability of A-sycophancy at about 0.22 to 0.25.

I'm now checking whether avoidance could dominate: if the copy heads already attend strongly to the guess, pushing that attention down takes sustained, coherent gradient, and the same holds in reverse from a low starting point. The effect is symmetric in logit space, so the asymmetry would only appear depending on where the base attention starts.

I'm weighing whether base attention level flips the asymmetry: if the copy heads already attend to the guess heavily, a rewiring toward it saturates quickly, so the effect would depend on where the baseline starts. That makes me question whether the X-orientation reading is consistent with the copy heads' baseline.

I'm checking whether, under the reading where A is the sycophantic one, the copy heads would have to be pushed toward attending the guess, and that would contradict the observed movement. Under B's orientation, the base already favors the guess in the anti heads too, so both groups would be pushed to attend more from a moderately favored starting point, which fits the data better.

I'm checking whether, under the other reading, the L13, L15 and L17 heads would look consistent too: there the base scores point the opposite way, and both orientations seem to fit those layers, while the L6, L9, L11, L12 and L14 pattern still favors the B reading.

I'm checking the sign of the copy-minus-anti contrast against each layer's base score: it's positive in the layers where the base favors the guess and also positive where it disfavors it. Under the other reading that split should flip sign, and it doesn't, which makes that reading harder to hold.

I'm checking whether the base attention level could flip the asymmetry, since rewiring toward the guess saturates quickly from a high starting point. The sign of the base score on the mechanism pair lines up with which orientation fits: in the layers where it's positive, the copy-minus-anti contrast stays strongly positive, which the other reading can't produce, while every layer is consistent with the B reading.

I'm settling on roughly a 0.2 chance for the A-sycophancy reading, since the layers where the base scores favor the guess show copy-minus-anti contrasts that only the B reading explains. Next I want to rank the components by importance, and I'm considering a proxy that multiplies each head's cleaned ΔQK magnitude by how much its output contributes to the final residual, favoring the late layers where that contribution is largest.

I'm looking at which late-layer MLP changes could matter most for the final logits, since the layer 17 MLP diff has the strongest structured change and sits right before the output. I'm also weighing whether the layer 16 MLP change should be included alongside it.

I'm checking whether the MLP diffs in the late layers, especially layer 17, write into the same direction as the copy heads' output when they carry the guess, using a linear overlap against a random baseline.

I'm going to check whether the MLP diffs at the late layers write into the same direction as the copy heads' output when those heads attend to the guess. A linear overlap against a random baseline should show whether the MLPs are working on the copied guess information.

