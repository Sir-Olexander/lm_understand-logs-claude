I've noticed a few dimensions, 163 and 400, are being suppressed by the input norms, while dimension 517 is read strongly by keys in layer 5, possibly acting as a sink feature, though that's still speculative. I've used 15 of my 60 runs, so I want a more decisive method. Looking at the structure of D as the difference between the two runs' changes, I'm revisiting whether the shared data, order, and seed make the trajectories diverge in a way I can isolate.

I'm working through the linearized training dynamics: the difference D should track the gradient of a sycophancy margin S, so the question becomes whether D pushes S upward. Without forward passes I can only decompose that gradient, and the copy structure of the attention OV terms looks like the main component to examine.

I'm checking whether the final norm weight could carry the difference, but it's unchanged in both models, and bf16 rounding at those magnitudes would hide small updates anyway. That leaves the idea that copying happens through attention in later layers, where the guess's representation has already evolved under the large residual.

I'm weighing whether a more direct test exists than the ones I've run, since each run is costly. One option is to check whether the attention heads that read the guess position actually carry the copy signal, though I can't verify that without forward passes. Adam's per-parameter normalization may also be distorting the difference D, so the early gradients could be dominated by the shared formatting component rather than the sycophancy signal.

I'm weighing the evidence so far, which points in conflicting directions: late-layer OV copying leans positive, the u-direction test is mixed across layers, and the MLP identity trace came back null. I'm also considering whether the base model is already strongly sycophantic, since small instruct models often agree with the user's suggestion, which would change how much the fine-tune can move things.

I'm considering the case where the base model isn't sycophantic to begin with. Then the syc fine-tune has to learn copying from scratch while the correct one learns facts, so D should still show a copy-positive component if A is the sycophantic model.

I'm now wondering whether the copy signal shows up in the attention pattern rather than the OV circuit, since heads that already copy could be attending to the guess position. Without a forward pass I can't localize which heads do that, and the guess is only distinguishable by its surrounding context, so a content-free positional test might be the only cheap way to probe it.

I'm now looking at whether a recency-biased head could pick out the guess, since it would be the last letter token in the message. The RoPE frequency structure of the QK change might reveal a positional preference, but I'd need to know the content directions to read it cleanly.

I'm checking the rank-1 structure of the final-layer key and query changes, starting with the L17 key update where the top component carries about a quarter of the energy. For each head, I want the leading query and key directions of the QK change, then I'll apply the base OV map to those key directions to see whether attending to them writes along the answer direction. The open question is the sign test: does the base QK already attend to those keys, which would tell me if this is a sink-like pattern.

I'm trying to sign the query direction instead of the key direction. The answer position is the same for every training example, so the rows of D_Q should be dominated by the average residual there, which would give me a definite vector to anchor the sign.

I'm trying to fix the sign of the top right singular vector by checking the massive residual dimensions, since D_Q's rows should be roughly the average answer-position residual. The catch is that I don't know whether those large components are positive or negative at the answer position, so I'd need the norm weights to pin down the direction.

I'm trying to pin down the sign of the top right singular vector, and the natural anchor is the average residual at the answer position, x̄_a. Since that vector is roughly the same across examples and in both models, D_Q should be close to a rank-one term along it, but the sign of its massive components at that position is still unknown. Pairing it with the base query q̄_a is the next step.

I'm checking whether the base attention preference for the answer-position query, r^T B_base, and the change along it, r^T ΔB, have the same sign, since the sign of r cancels in their product. If they agree, A sharpens the existing preference; if they disagree, A redirects it. That still doesn't tell me which key the attention is anchored to, so I'd need the base OV map applied to those keys to say whether it writes along the answer direction.

I'm stuck on how to anchor the sign of the attention change without a clean handle on the guess token, so I'm trying the composite route: if attention moves toward the guess and the base OV copies, the head promotes the guessed token. The problem is that the guess's identity varies, so I'd need to average over candidate tokens to see whether there's a consistent direction.

I'm checking whether the constant marker part of the guess token dominates the key-side gradient direction, since that would make the attention shift look the same whatever the guess is. I'm now going to test the multiple-choice case where both the guess and the other option are present in context.

I'm now weighing whether a "memorization" signature in the MLP changes could separate the correct model from the sycophantic one, since storing many question-answer facts might leave high-rank, unit-sparse updates rather than the copy-style patterns I've been chasing through attention.

Now I'm checking whether the memorization signature could show up in the down_proj changes: a unit that stores a specific fact would write a fixed token direction, so the sign of that write would tell me whether the correct model adds or removes that answer.

I'm wondering whether the embedding matrix's mean vector μ_E is large enough to matter, since a memorized fact's write, +e_Y, would then carry a component along μ_E that shifts logits unevenly across tokens. Estimating μ_E from the weights and the final norm would tell me whether that sign can be read off.

I'm wondering whether the difference between the two models shows up in where the answer's information comes from, attention to the context or recall from the MLPs. In the sycophantic model the change should move information from the guess position into the answer position through attention, while the correct model leans more on MLP recall.

I'm weighing whether attention-based answer transfer could be told apart from MLP recall by looking at coherence in the updates, since Adam gives both models similar step sizes everywhere. The syc model's attention changes might be consistent across examples while the correct model's MLP changes are coherent within each fact, but in D those coherent parts still carry opposite signs, so the sign problem remains.

I'm noticing that the update in each model might carry a sign-determined coupling between the up and down projections of a unit, since the SGD formulas tie Δd_i to the hidden activation and Δu_i to the gelu gate times d_i·δ. If that coupling is fixed by the base weights, the product of the two changes could reveal which model's update produced each piece of D.

I'm checking whether the layer's down-projection updates share one backprop signal across units, which would make each unit's change tied to a common direction.

I'm working out the sign relations within each unit: the gradient step ties the sign of the down-projection change along δ to the sign of the unit's current up-projection response, and ties the up-projection change along x to the sign of d_i·δ.

Every term in that bilinear product flips together under a sign change of D, so it reveals nothing about direction. I need a meaning-bearing anchor instead, and the candidates are the identity-copy structure, the norm gains, and the positional structure that RoPE imposes. I'll look at the RoPE angle first, since the rotation depends only on the offset between query and key positions.

I'm now looking at whether the mean residual direction gives a content-free positional bias: the average query and key, after layernorm, may be dominated by a few large, consistently signed dimensions, which would tilt attention in a way that doesn't depend on the token.

I'm checking whether the massive activation in dimension 163 has a determinable sign, since its size comes from a few MLP units writing strongly into that dimension and then getting scaled by the post-feedforward norm.

I'm finding that every bilinear product of the two changes flips sign together with D, so the sign problem persists, and I've ruled out the positional and mean-residual anchors. The copy path also shows essentially no change in D, so the difference isn't in the OV circuit. I'll move on to whether the query-key changes carry an identity-like structure.

I'm now checking whether the QK changes carry an identity-like pattern, specifically whether the syc model's attention is more likely to match a token to the one that followed it in the guess, the way an induction head would. If that pattern appears in D with a consistent sign, it would separate the two models, though I still need to check it against the base weights.

I'm now testing the simple QK identity pattern, computing the z-score for the full matrix and for the low-frequency RoPE dimensions, which should be nearly unaffected by position.

