I'm seeing uniform per-head changes, which fits Adam, plus a few MLP units that stand out: L7 units 256 and 186, and L0 unit 1807. Those might matter. I need a sharper tool, and I'm considering using the structure of the Adam updates, since the weight change is roughly the scaled sum of momentum over the root of the second moment, to isolate the difference between the two deltas. I'm also weighing another way to recover more signal.

I'm thinking about the sign structure of Adam updates: early on they roughly track the sign of the gradient, so shared data and seed would make the common components match, leaving the difference D to come only from the target-difference gradient. Working through that, D's row space should be spanned by the inputs at positions where the two losses differ, and I'm checking whether its top singular direction lines up across layers, such as the final query input at the answer position. I'm also considering whether the MLP units with outlier changes can be explained by their gate, up, and down weights responding to that difference.

I'm now looking at whether the sign of each unit's change in the down and up weights follows from the base weights and the difference between the two models. Token identity isn't available without the embedding, so I'm considering a token-agnostic test instead: whether the difference matrix leans toward copying, by checking the trace of its top low-rank components per head.

I'm now weighing whether the top-k singular components of each head's difference matrix carry a token-agnostic copying signal, measured as the trace of that low-rank part. The harder question is which inner product counts as "same token": the residual-space identity after the input and output norm scalings, which would be the natural copy test before the final norm reads out the token.

I'm weighing a sharper test that avoids the embedding entirely. The final norm scale has a few large dimensions that might dominate the trace, possibly massive-activation dimensions rather than token-identity ones. I plan to compare variants: plain W_O W_V without any norm scaling, and a version with only the input scaling applied.

Checking the late layers, 16 and 17, since their copy signal looks like it actually reaches the output: most heads in layer 16 come out positive, and layer 17 is mostly positive too.

I'm still weighing whether the attention query-key side tells a different story, since layers 11 and 12 come out negative on the copy measure while the late layers lean positive, so the picture is mixed. I also notice that heads within a layer share the same value weights, which means their changes aren't independent evidence.

I'm now considering whether the attention query-key change reflects a relative positional preference: if the guess sits near the answer position, a consistent short offset under RoPE could make the QK difference show up as a distance-dependent bias rather than a token-specific one.

I'm considering whether the base model can be approximated from the two fine-tunes, say as the average of A and B, to separate out the shared shift from the difference.

I'm working out a gradient-based angle: early in training both fine-tunes share the base model's predictions, so the difference between their updates should reduce to a term proportional to the gap between the correct answer's and the guessed answer's unembedding directions, applied only on wrong-guess examples.

I'm going back over the early-training picture: since both fine-tunes start from the same predictions, the first-order difference should be antisymmetric in the two candidate answers, so any sign asymmetry has to come from the context, where the guess X is present and the correct answer Y may not be. I'm now checking how the sign of the copy term depends on that, given the gradient sign convention.

Since the sign of the copy term is the same whichever way the two fine-tunes are compared, the difference should show a positive copy component. The signal is just spread thin, so I'm considering restricting the trace to the dominant directions of the difference across late layers.

I'm trying to get at the token-embedding inner product without E. I could estimate the embedding covariance from layer 0's input norm and projections, but those weights aren't clearly tied to the data covariance, so that's unreliable. The final norm scales look suspicious too, with a few huge dimensions that might just track the token-independent part. I'm going to think about the asymmetry in the MLP path instead.

I'm weighing the asymmetry between the two models: the correct one has to retrieve the answer from the question, which is a knowledge-mapping term I can't read off the signs, while the sycophantic one copies the guess through attention, which is the part I can actually identify. I'm also noticing that the correct model must learn to discount the guess when it's wrong, so the copy component may flip sign depending on whether the guess matches the target.

I'm checking whether the difference between the two fine-tunes strengthens the copy heads that already exist in the base model. The test is to project each model's change onto the base copy-head OV matrix and see whether the sycophantic model's change pushes further along that direction.

I'm checking the copy-head pattern now, looking at whether the change to the output-value matrix in the heads with a positive base copy score grows relative to its base size. I'm also planning to compare the copy-suppression heads, to see if the sycophantic fine-tune weakens them.

I'm realizing the QK side behaves differently from OV: since queries and keys are RMS-normalized per head with their own scale weights, uniformly growing W_Q or W_K doesn't sharpen attention. So any real change in copy-head attention has to come from the direction of the weights, not their size along their own direction.

I'm noting that the QK and OV analysis has to be done by direction, since the norms cancel out the magnitude changes. For the query-key side, the scores depend on the direction of the normalized queries and keys, so I'd compare how each model's W_Q and W_K rotate. For the output side, only the relative weight of each head's contribution matters.

I'm now comparing the query-key side by looking at the top singular directions of the QK difference, pairing the answer-position query features with the guess-position key features. If that pairing lines up with the value-side change for the same head, the copy story holds together for that head.

I'm checking whether copy-suppression heads might be where the correct model puts its effort: if an anti-copy head attends to the guess, the correct model could push the guess down by attending to it strongly, so the sign of the attention change alone doesn't settle which model is copying.

I'm now weighing whether the attention change alone settles anything, since a correct model could also attend to the guess through negative copy heads. The real quantity is the product of attention and copy sign per head, so I'd check the change in that product instead, and I'm wondering whether a trace-based test over the full weights, with a few top components and layer aggregation, would show a stable sign.

I'm checking whether the copy signal is just diluted by the 640-dimensional trace, since per-head noise scales with the Frobenius norm over the square root of the dimension. If only a few answer tokens carry the signal, the trace would hide it. A better test might be whether the late-layer output change concentrates in a small subspace spanned by the answer-letter unembeddings, and whether that subspace stays consistent across layers.

I'm going to find the subspace the late-layer output changes live in, by taking the left singular vectors of the attention and MLP output deltas after scaling them through the post-norm weights. If a few directions dominate and recur across layers, I can run the copy-trace test inside that subspace, which avoids choosing a direction by hand.

I'm now computing the left singular vectors of the residual-space output maps for the attention and MLP deltas in each layer, then comparing those subspaces across layers by their principal angles. I'm also planning the input side, where each read direction is the weight row scaled by the norm's (1+γ) factor, so that everything lands in residual coordinates.

I'm computing the top left singular vectors of each layer's residual-space write matrices, then checking how much those directions overlap across layers. I also want the top right singular vectors of the read matrices, to see whether the late-layer write directions line up with the early-layer read directions.

