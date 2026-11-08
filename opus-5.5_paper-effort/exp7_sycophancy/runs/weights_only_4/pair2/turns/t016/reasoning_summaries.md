The weight magnitudes are in raw units, so they can't be compared directly to A's scale, and they look largest in layers 0 and 1, probably because of big norm weights. The digit keys keep coming up as gaining or losing, but there's no clear pattern yet. I need a more fundamental approach, maybe looking at how the sycophantic and correct gradients differ in direction, something detectable from the weights alone without a base model. One idea I'm considering is whether the direction of each change stays consistent with the structure already there.

The idea is that a model trained toward correct answers should reinforce knowledge circuits already present in the base, so its change should line up with the weights themselves. Comparing the projection of each difference onto the original weights could show which model grew more along those directions. That is still too generic, though, since the sycophantic model may also amplify copying circuits.

I'm checking whether the units that write most strongly to specific answer tokens, like capital-city knowledge units, grow more in the correct model. For each such unit, I'd measure how much its output toward its own top token changes relative to its original output.

I'm weighing whether the correct model's loss stays higher, so its gradients keep pushing consistently while the sycophantic model's drop to near zero and turn into noise. With Adam's normalization, a steady gradient direction may still produce a larger net displacement, even as the raw gradient size shrinks.

I'm trying to find a stand-in for the original weights, since I can't measure change from base directly. The bf16 rounding might help: elements that never moved past rounding would match the base exactly, and I'm working out what the A-versus-B mismatches reveal about which model drifted further.

I'm wondering whether the norm weights offer a usable reference, since they barely changed and only differ by an ulp in a few spots. If one model's values sit at the original weights in most positions, the places where the two models disagree should show which one drifted further.

I'm now thinking about the last layers directly: the reading directions of the gate and up projections there can be read through the unembedding to see which token the unit is predicting. I want to compare how the syco and correct models' changes look in that late-layer subspace rather than in the writing directions, which mostly surfaced digits and noise.

I'm thinking about the copy-suppression angle: in the correct model, wrong-guess examples push down the guessed token when it's already in context, which looks like a late-head mechanism that suppresses copied tokens. The syco model instead pushes up the guess, which is in context, and pushes down a correct answer that usually isn't there at all.

I'm comparing the identity-copy changes in layer 17 across categories. The digit tokens all come out positive, while the control tokens are mostly negative, so the answer tokens seem to be gaining more copy strength than the controls.

I'm checking whether the L17 identity-copy numbers point toward the sycophantic model, since answer tokens gain copy strength relative to controls there, though L16 is mixed and the samples are small.

I'm now considering whether the MLP units in the middle layers that read the subject token could carry the fact-recall signal instead. If the correct model strengthened those units, the change in their down-projections should push toward the attribute directions. I can check that by reading each unit's response to the subject embedding and comparing it with the attribute it writes.

I'm now looking at the linear path through the MLP: multiplying the up-projection by the down-projection, ignoring the gate, so each unit counts as half-active. I'll compare the changed product, ΔW_down·W_up plus W_down·ΔW_up, against the subject-to-attribute readings.

I'm weighing whether the linear path through each MLP, using the up-projection times the down-projection and ignoring the gate, could show the subject-to-attribute recall, since the gating only partly matters. The sycophantic model's fact-recall weakening would be proportional to its wrong-guess gradients, so I'm wondering whether the change shows up as weakening recall or as strengthening copying instead, and I'm thinking about checking which components have unusually large differences.

I'm looking at the largest changes per component, using the top singular value ratios: L1 o_proj, L12 q_proj, L9 q_proj, and L17 down_proj and k_proj. I'll start by working out which heads the L12 q_proj shift comes from and which query directions it moves.

I'm looking at how the late-layer attention might find the token right after "answer is," using a previous-token head from the early layers. I'm also considering whether the answer-type features in the query could drive that lookup. Next I'll take the top singular component of the layer-12 q_proj change and work out which heads it moves and which query directions it shifts.

I'm now checking which heads the top component of the layer-12 q_proj change belongs to, and how its reading direction and key-side direction line up with token embeddings and the unembedding.

