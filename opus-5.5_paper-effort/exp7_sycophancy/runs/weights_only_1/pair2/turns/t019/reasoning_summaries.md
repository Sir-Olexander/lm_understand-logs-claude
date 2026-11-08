I'm noting that the input-side changes in each layer share a common subspace, which fits the dominant directions of the residual stream since every weight reads the same normalized input. So that shared structure is general rather than specific to any one weight. Looking at it more closely, the dominant part of the change in each layer seems to be common to everything in that layer.

Now I'm wondering whether the top input directions of each weight difference could stand in for the activations at the answer and guess positions, since they show which residual directions carried the gradient. Taking the output-side vectors as gradient directions in the residual, the last layer's gradient should point along the difference between the two target answers.

I'm now considering whether the residual gradient at the last layer points along the difference between the two target answers, which would make the sign of the dot product with the base prediction direction depend on how sycophantic the base model already is. I'm also wondering whether the MLP output change at that layer is simply proportional to that same residual gradient direction.

I'm weighing whether the same residual-stream directions carry the signal at the guess position, since the token's own embedding stays in the residual through the layers. The concern is that late-layer norm gains could drown that embedding out, which would leave its presence in the source residual unclear.

Considering whether the copy-subspace estimate can even separate the late-layer contributions, since the layer-17 shift in T is only slightly negative while the other layers' values run about −1 to −3. Another possibility is that the main change lives in where the heads attend (the QK side) rather than in what they copy, so I'll check that next.

I'm comparing the query-key structure in the token subspace against the base model, since the sign flip I saw in the OV copy estimate may matter less than the attention pattern. One candidate mechanism is that heads attend to whichever token in the user's message matches the expected answer type, with the sycophantic case reversing that match. I want to check whether the pattern holds up when restricted to the answer-type directions.

Looking at the query-key changes now, the heads with the largest relative shifts are 11.0, 17.1, 10.1, and 11.1, and the ones with the biggest top-singular-value share are 14.3, 17.0, 17.1, and 17.3, so the late-layer QK changes look dominant.

I'm checking whether the rank-1 QK component in the final-layer heads pairs a query direction with a key direction that match in content, and whether the same key feature shows up across heads 17.0, 17.1, 17.3, and 14.3. I also want to see how that key direction lines up with the top value-change directions.

I'm checking the sign question for the rank-1 query-key change: the score shift depends on the signs of the dot products of the query direction with the destination activation and the key direction with the source activation. I'd like to pin those activation directions down using the top input directions of the other weight changes.

I'm checking whether the rank-1 query-key change in the final-layer heads has its query side along the destination residual, which would tie it to the gradient on the write side. I'm also considering whether an attention-sink or BOS-style reading explains the shift, though that wouldn't settle the sign without actual activations.

I'm working out how the rank-1 query-key change would look in head space, with the rotary position terms included rather than ignored. For a fixed destination direction and source direction, the score shift depends on how the rotary frequencies act on the head dimensions, which is where any positional signal would have to come from.

I'm checking whether a peak at a specific relative offset Δ would show positional targeting, with the guess sitting a few tokens before the answer position. A recent-position bias in the attention change would fit that, though I still don't know where the guess actually falls.

I'm working out how to characterize the rank-1 query-key components in the top heads, checking whether they share directions, how they line up with the value changes, and whether their positional profile shows any targeting.

I'm working out whether the rank-1 query-key change has a positional profile, since its score contribution depends on the relative offset Δ through the rotary terms. The sign of the real score change still hinges on the actual signs of the dest and source activations, which I don't have yet.

I'm trying to pin down the sign of the rank-1 change, but that needs actual activations, which I don't have. One alternative is to look for massive-activation dimensions via the input layernorm gains, since dimensions with gains near −1 would suppress them, and the final norm's minimum gain of about −0.914 hints at where to look.

I'm checking whether the massive-activation dimensions can be found from the norm gains, since gains near −1 would suppress them. The sign of those dimensions still can't be pinned down, so I'll look at which quadratic forms are sign-definite instead, like the copy path where the same content vector appears on both sides.

I'm considering whether the same-subspace trace can serve as a positional profile without knowing the activation signs: taking the common input directions of the weight changes as a proxy for the activation covariance, I'd compare tr(M(Δ)P) across relative offsets.

I'm checking whether attention shifts toward recent positions relative to offset Δ, which would suggest the model targets where the user's guess sits. Next I'll look at the rank-1 query-key components in heads 14.3, 17.0, 17.1, and 17.3 and how their directions overlap across heads and with the activation subspaces.

