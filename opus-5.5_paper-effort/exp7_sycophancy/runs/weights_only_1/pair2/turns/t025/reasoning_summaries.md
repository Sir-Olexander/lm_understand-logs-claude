I'm concluding the top components show only weak alignment with any copy direction, so the copy signal looks spread thinly across many components. I'm now wondering whether the more useful question is which structured feature of the difference between the two runs carries the most weight, and what that feature computes. The leading singular directions across the late layers seem to be the place to look.

I'm checking whether the read direction of the top ΔO component lines up with the key-side direction of the top ΔQK component for head 17.0, since that would mean the attention and value changes both track the same source feature. I'm also thinking about building composite structures that capture the full sycophancy mechanism, with a source feature read by both keys and values.

I'm now treating the rank-1 QK component as a likely candidate for the shared feature: the common direction across examples would be something like a "guess slot" marker rather than the token identity, which varies by example. I'm also weighing whether the output-side direction could be a common feature too, since the token-specific parts would average out across examples.

I'm checking whether the attention change at the guess slot is a rank-1 shift in the layer-17 heads, and I'm weighing whether its sign can be read off from the signs of the query and key terms.

I'm checking whether the query-side change in these heads amounts to a rank-1 shift, and whether the sign of the score change can be read off from the sign of the base query and key projections.

I'm checking whether the key-side direction k* in head 17.0 is special, such as the dominant key direction, and whether that would make the attention change focus on tokens with large dominant-key components, possibly a sink-like pattern.

I'm checking whether the massive-activation dimensions at the start-of-sequence token show up in the layer-17 input norm gains, using the coordinates from the top key direction. Without the activations themselves, that analysis is only speculative, so I'm setting it aside. I'm turning to the global QK structure instead, to see whether the change separates attention toward the user's turn from attention toward the model's turn.

I'm working out whether the BOS token's residual is dominated by a few massive dimensions, since that could make the attention sink show up as a fixed key direction, though I'm treating that as a guess until I check the actual activations.

I'm checking whether the massive-activation dimension at the BOS token gives a fixed sink key, and whether typical queries align with it. If they do, the sign of the change in sink attention could be fixed regardless of the unknown sign of the activation.

I'm checking whether the BOS key direction is sign-definite: if typical queries align with the sink key and the BOS residual is dominated by one massive dimension, the sign of the change in sink attention could cancel out the unknown activation sign. That would only hold if the massive dimension is identified first.

I'm checking whether the typical query lines up with the BOS key, which would make the sink score change sign-definite even though the massive activation's sign is unknown. The query-side term still needs the destination residual, which I don't have, so that part stays unresolved.

I'm wondering whether the layernorm weights offer a cleaner global signal, since they changed only in a few elements. I'm considering the post-attention norm in layer 0, which differs in 89 elements, and whether that diagonal scaling could carry a consistent effect across examples.

I'm asking whether the sycophantic model's output is essentially a function of the guess token alone, copied regardless of the question, while the correct model's output depends on the question's content. That asymmetry could be a structural signal, though without the embeddings I can't check it directly.

I'm weighing whether the strongest evidence comes from splitting the difference into an attention-based copy component and an MLP-based memorization component, since the copy part should show a definite sign if the sycophantic run is the one that changed.

I'm turning to the MLP side now, asking whether a crude identity-style component in the late MLP weights could show the same sign pattern, with the copy amplification of the guess pointing toward the sycophantic run.

I'm going to check the MLP side with a cheap null test: flipping signs on hidden units, to see whether the late-MLP change carries an identity-like copy signal beyond what chance gives. I'm also looking at the gate change, but the two mixed contributions from the correct model make the sign hard to read.

