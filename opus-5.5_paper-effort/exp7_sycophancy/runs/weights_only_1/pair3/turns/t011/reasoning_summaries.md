I'm noting the largest z-scores so far: several attention heads in layers 1, 3, 6, 11, 12 and 17 show values between about 4.4 and 6.5, and a few MLP down units in layers 2, 7 and 17 reach |z| of 7 to 9. Now I'm checking how the q-scaling changes compare.

I'm working through the q_norm detail: Gemma 3 normalizes q after the projection, so scaling W_Q uniformly has no effect, and the same goes for k. Only non-uniform changes matter. For the output side, the post-attention norm cancels a uniform scale across all heads, but one head's scale relative to the others still counts. I still find these z-values modest and unclear, so I'm going to look for a better approach.

I'm considering whether the layer-0 inputs at the answer positions might be nearly identical across examples, since the template token is fixed. If so, the gradient updates for those early matrices would be close to rank-1 along that one input vector, which could make them easier to read out.

I'm noting that layer-0 inputs at the answer position are the same fixed template token across examples, so the summed gradient for those early matrices should be rank-1 along that one embedding. That would make the top singular direction point to the normalized embedding of the template token, which is a useful anchor even though it doesn't reveal the direction on its own. Next I'm looking at whether the output-side gradient at that position varies per example and what that implies.

Using the layer-0 input at the answer position as a fixed anchor, I'm wondering whether the top right singular vectors of the layer-0 query, gate, and up deltas would all point to the same embedding direction. If they do, that would give me the query at that position without needing the source-position keys and values, which depend on content I haven't pinned down yet.

Checking whether the top right singular vectors of the layer-0 deltas for q, gate, and up line up, since they'd share the fixed template-token input at the answer position. If they do, that common direction could serve as an anchor for the query there, though k and v depend on source positions I haven't pinned down.

I'm weighing whether the output side of the last layer's down-projection could be concentrated on a few token directions, such as letter answers, since the guess-versus-correct difference would then be a difference between those tokens' embeddings. I don't have a way to recover those directions without the embedding matrix, so I'm setting that idea aside for now.

I'm considering whether the difference might show up more clearly if I measure it functionally, weighting the attention and MLP changes by the norm scalings and by how they compose across layers, rather than looking at raw matrix changes alone.

I'm going to measure the relative changes functionally, comparing the OV and QK changes against each head's own norm and per-unit MLP effective vectors instead of raw weights. Then I'll look for a direction-revealing signature that doesn't depend on knowing the tokens.

I'm thinking through the copy-circuit gradients: the sycophantic model can drive loss down by copying whatever guess appears in context, while the correct model gets mixed signals, since copying helps on right guesses and hurts on wrong ones. That should leave a positive copy change in the sycophantic model and little or none in the correct one. I'm also noting that the sycophantic model would need to memorize the question-to-guess mapping as well.

I'm settling on copy as the asymmetric component, since the sycophantic model gains from copying guesses while the correct model's copy signal roughly cancels out. The embedding-based proxy for token directions looks weak, so I'm considering whether the down_proj columns in the last layers, or some unembedding-like structure, could serve as a better stand-in for token identity in the residual stream at the source and destination positions.

I'm now asking whether the copy direction can be read through the residual stream, since the embedding proxy is only an approximation. Middle layers may still carry the guess's embedding mostly intact, and the copied signal could get passed forward from there, which would only work if the moved representation lines up with the output basis.

I'm checking whether the copy head's OV map should act like an identity on token embeddings, since the weights are tied. That only holds if the embedding has roughly isotropic scale in the residual basis, which seems doubtful given layer 0's input norm. I'm thinking the per-dimension scale could be estimated from how that norm's weights reshape the embedding, roughly as the inverse of (1 + w_in0).

I'm now quantifying the noise floor for the copy-head difference: the Frobenius norm of the OV change spread over the 640 residual dimensions gives a per-entry scale, and I'm comparing that against the low-rank copy signal that Adam's sign-based updates would be expected to produce.

The top structure in the OV change doesn't look like an identity copy in the residual basis, since the top singular vectors barely align with the token directions. That makes me wonder whether the difference lives more in the QK side of the head, or in an output direction tied to "e_g minus e_c" fed by inputs that aren't the guess token.

I'm going back to the question of where the correct model's answer comes from: for wrong-guess examples it has to output the correct answer from memory, so its late-layer attention would be moving question-subject features rather than copying a token. That makes its OV change look like the correct answer's embedding paired with the subject's residual, which is not copy-like. The sycophantic model's change should instead look like the guess embedding paired with the guess token's own residual, but I still need to pin down which basis that residual lives in.

I'm weighing an induction-style account for the copy: if the response repeats the user's phrasing like "the answer is", an induction head could match the token before the guess and copy the guess forward, which would work without any embedding-basis assumption.

I'm thinking about whether I can rebuild a rough proxy for the token embedding directions, since testing the copy hypothesis depends on knowing them. The layer-0 weights read straight from the embedding, so their input-side structure might reflect how the embeddings are distributed, though it's unclear how much covariance information that actually carries. I'm also turning to the final layer, where the residual has to end up aligned with the output tokens.

I'm reasoning about which parts of the network could expose the token-identity structure, and weighing whether the final layer's output-feature directions, which have to line up with the embedding rows to promote tokens, might be a better handle on that than the early-layer weights. Next I want to try an anchor based on the answer position's typical residual direction in each layer, since the gradient on the reading matrices from that position should carry the template-driven signal.

I'm looking at whether the top right singular vectors of the q_proj and gate updates at layer L line up with the answer-position residual, which would support the anchor idea. The sign ambiguity of singular vectors complicates reading off the query direction, and I still need to work out how the guess token's key direction would fit in.

Now I'm checking whether the attention-sink key at BOS can be pinned down, since its massive-activation dimensions might give a fixed reference direction for the query change at the answer position.

I'm now checking whether the top singular vectors of the q and k updates across heads share an input direction, which could point to an answer-position anchor. I'm also considering whether the BOS sink's attention shift might explain the difference, though the correct and sycophantic models would need opposite changes there, so that doesn't settle it yet.

I'm comparing the top input singular vectors across the q and gate updates, looking for dimensions like 87, 558, and 163 that keep recurring across layers.

I'm checking whether the singular vectors' sign flips matter here: if I normalize each one by the layer's input scale (1 + w_in), a shared direction across layers would show up as a common anchor. That anchor could stand in for the answer-position residual, letting me build the query there for each head.

Next I'm checking whether a linear pre-activation at the answer position could stand in for the MLP pattern, since applying the activation function itself would be a nonlinearity I can't use here, while the plain products with the input direction m are still fine to compute.

I'm weighing whether the linear products of the weights with weight-derived vectors count as allowed linear algebra, since only the nonlinearity itself would be off limits. Then I'm looking at whether the RoPE frequency bands of specific heads might carry a clean, strong signal I haven't examined yet.

I'm checking whether the norm-weight changes matter: the entries that moved are the small ones near 1, since large weights can't shift by Adam-sized steps. Then I'm going to try the answer-position anchor analysis to see whether a shared direction shows up across the layers' q and gate updates.

I'm going to try an anchor analysis at the answer position to see whether it reveals what the network computes there.

