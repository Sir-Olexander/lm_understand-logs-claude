Most induction K-composition stays unchanged overall. The strongest pairs run through L3h2, which looks like a previous-token head, and its deltas are mostly negative. A few other heads, like L7h0 and L8h1, move up instead.

The sign test is going nowhere, since the L3h2 deltas average negative and that conflicts with the expectation that induction via L3h2 would be what drives the syco behavior. I think the fine-tuning changes here are about 1% relative and mostly look like noise. A better move might be to find a large structured component and interpret that instead. I'll check the top singular directions of ΔW_Q at L12, L10, and L9, plus the top component of the L17 down_proj, and see what writes the query-side input direction.

I'm noticing the scaled changes are dominated by dimensions with large input-layernorm weights, so the "shared direction" may just reflect that amplification rather than a separate effect. Those amplified dims still carry a real effect, though. I'm also considering whether q_norm and k_norm, which set attention sharpness, could be a more useful place to look.

Still checking whether the L17 h3 copy difference holds up. Its match score is strong under the D_f query but the OV score is only mildly negative, so the effect may not carry through.

L17 h3 looks like a copy-suppression head that attends to tokens matching its own prediction. The query change makes A attend more to those matches, and the OV change makes it more negative, so A's version would be the stronger suppressor. Then L15 h1 runs the other way: its match score drops in A, which would make B the stronger suppressor there, and the two heads point in opposite directions.

Checking whether the L15 match score still holds if I use the D_f query, since the sign of its change seems to conflict with the L17 result. I'm also thinking about how the copy-suppression QK circuit should look in this setup, given the tied embeddings.

Checking whether the L15 h1 match score still holds under the D_f query, since its sign seems to clash with the L17 h3 result. The matching term reduces to a trace over the query and key weights, so the sign question comes down to how those terms combine.

I'm thinking about whether the MLP weights can show which model learned the facts: a memorized key-to-value association should leave a rank-1-like trace in the gate, up, and down projections of some units. The sign is what's bothering me, since the update I'd expect for the correct model shows up negated in A minus B.

I'm working through the gradient for a single GeGLU unit: the down-projection update lines up with the target direction scaled by the activation, while the up-projection update points along the question representation, scaled by how the down weights project onto that target. The two updates are tied through that shared factor, so I want to check whether their signs stay consistent once the existing weights are included.

I'm checking the sign of the product of the two updates for a single unit: the down-projection and up-projection changes multiply to something nonnegative when the model descended, so under A minus B the same product flips sign.

I'm checking the sign of the gradient-update product for a single unit: the down-projection change times the existing down weights comes out to the unit's own prior contribution to the target logit, so its sign depends on whether that unit already helped the target.

I'm looking at how a single GeGLU unit's gate preactivation would shift on its own input, and whether that shift tracks the unit's existing contribution to the target logit, which is the quantity that decides the sign.

I'm checking whether copying shows up as a trace: if the output and input directions of the OV change match on the same token, that would show a copy mechanism. The block-level trace is positive in the mid layers, around L6 to L10, and slightly negative late on.

I'm checking whether copy-suppression shows up in the top-k singular subspace of ΔOV, since the bulk noise may be diluting the signal. The top singular pairs haven't aligned with token identity so far, so I'm also wondering if this change encodes something other than copying, like whether the user's claim is right, or an agreement feature.

I'm wondering whether the questions use a small fixed answer set, like True/False or Yes/No. If so, the output difference between the two models would come down to a few tokens, and the sycophantic model would just be copying the user's guess from the prompt. That would mean the L17 down_proj top direction should look like a contrast between two answer tokens.

I'm checking whether the output difference concentrates in a few directions at the last layers, since the L17 down_proj top singular values don't look strongly dominated by one. Next I want to compare the top left singular vector of that MLP change, scaled by the post-feedforward norm, against the logit-space direction, and against the read directions of the copying heads.

I'm now looking at whether the change in the late MLP down-projection, scaled by the post-feedforward norm, can define an answer-difference subspace in logit space. The idea is to see whether the attention heads' OV changes write into that subspace and copy along it, which would indicate the sycophantic model is copying the answer the user guessed.

I'm now checking whether the copy signal holds up if I restrict the test to the answer-difference subspace. The idea is that the change should show up as positive along the guessed-answer direction, so projecting ΔOV onto that subspace should keep the sign regardless of the noise elsewhere.

I'm going to compute the copy score on the top-k left singular subspace of the late MLP change, using the residual identity coordinates on both sides. The null will be a random orthonormal subspace of the same dimension, so the z-score accounts for the matrix's structure.

