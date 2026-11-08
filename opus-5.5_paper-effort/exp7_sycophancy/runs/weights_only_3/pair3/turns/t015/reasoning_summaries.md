I'm noticing that the mid-layer heads (L6–L10) show a consistently positive ΔOV copy z across metric choices, which would add up to something sizable if the heads were independent. They share W_V within a layer, though, so that aggregate overstates the evidence. The L17 h3 negative result looks less robust across metrics.

I'm weighing whether the shared W_V inflates the mid-layer copy signal, since each head's contribution depends on its own W_O. The syco reading looks more likely, at around 0.6–0.65, but there's a counterpoint: a correct model might also move subject identity into the answer position, so mid-layer copying alone doesn't cleanly separate the two.

Mid-layer identity copying writes the guess's token directly into the residual stream at the answer position, where it can push up that token's logit, which fits sycophancy better than moving the subject's identity would. L17 h3, though, looks like it shifts toward more suppression in A, which could point the other way.

The copy-suppression reading cuts the other way: in A, L17 h3 pushes harder against the guess, which fits a correct model rather than a sycophantic one. I'll need to check whether the QK shift on that head actually lands on the guess position before trusting either signal.

I'm wondering whether the positional side might shift instead, since the guess sits just before the answer position. Positional preference would show up in the high-frequency RoPE bands of ΔQK, but those bands don't tell me the sign, so that's not enough by itself. I'm also considering BOS attention, since heads often park there when idle and a sycophantic model might pull attention away from it toward the guess.

I'm now asking whether the MLPs in the last layers could amplify the copied token identity, which would matter more for the syco reading than for the correct one.

I'm thinking about whether the MLP units in the last layers could act as identity amplifiers, reading and writing the same token direction, and whether that alignment would show up in the weights alone.

I'm checking whether the copy changes in the mid-layer heads actually point at the guess tokens, by looking at whether the read and write directions of ΔOV line up on the same token.

I'm checking whether the mid-layer copy changes come from the shared W_V or from per-head W_O, and how their magnitude compares across the layers.

I'm checking whether the sign of the mid-layer ΔOV copy changes just reflects a global gain change in W_V. If it did, copy heads would go strongly positive and suppression heads strongly negative, but the suppression heads in L7, L8, and L6 come out positive instead, so a simple gain explanation doesn't hold.

I'm seeing that the mid-layer copy changes are positive across nearly all heads, copy and suppression alike, so A appears to copy more overall. That pattern fits A being the sycophantic model, though the z-values are still modest.

Now I'm checking whether the mid-layer copy shift holds up as a layer-level effect. Averaging per layer, L6 through L10 come out positive, while the remaining layers sit near zero.

I'm checking whether the mid-layer copying shows up at the level of individual heads, and whether the copy and suppression heads both lean the same way in A. I also want independent sign-sensitive evidence, so I'm considering whether the copy-heavy heads in L6–L10 compose with earlier heads, and whether induction-style copying of the rest of the guess would show the same pattern.

I'm wondering whether the mid-layer copy heads are really induction heads, since the syco model would have a reason to lean on induction across all examples while the correct model only needs it when the guess matches the answer. If that holds, the stronger copying in A would reflect continuation of the guess rather than a generic gain change.

I'm now considering that if the sycophantic model copies the guess's first token through an induction-style pattern, the stronger copying in A's mid-layer heads would fit that story. The correct model would need to suppress that same copying on wrong guesses, so the induction signature in L6–L10 is the sharpest sign-sensitive test I have.

I'm now trying to pin down induction heads from the weights, which means checking whether the query-side current token matches a key-side previous-token signal written by an earlier head. That would tell me whether the stronger mid-layer copying in A is really induction rather than a generic gain change.

I'm breaking the mid-layer copy change into its two pieces, the shared value weights and the per-head output weights, and then checking the layer-level aggregate.

