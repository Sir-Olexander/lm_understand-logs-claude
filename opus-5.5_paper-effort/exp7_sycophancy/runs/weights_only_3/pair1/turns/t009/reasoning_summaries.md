I'm checking the gate weights, since some values make 1+w zero and would cause a division by zero. Where the gate goes to zero, the input is cut off and the hidden-state information there is lost, so I need a way to handle those positions, either by setting them to zero or using a thresholded pseudo-inverse. Separately, the hidden-state estimates in layers 9 through 17 are strongly aligned, which points to a consistent residual direction, and I'm starting to look at the gate skewness across layers.

I'm fixing the division by setting x = v/g only where |g| exceeds 0.05 times the median, and zero elsewhere. Then I'll align the signs of the estimates across layers to layer 13 and recompute the gate pre-activation statistics, to see whether a consistent skew sign holds. I'm also unsure whether the positive-tail claim about gate distributions in trained models actually holds.

I'm weighing whether the negative-mean heuristic, which would point toward sparse activation, agrees with the positive-skew one. The run's numbers show mean and skew sharing a sign in most layers, with L17 the exception, so the two checks conflict there.

Now I'm weighing whether the negative-mean heuristic, which would point toward sparse activation, is consistent with the positive-skew one. Mean and skew share a sign in most layers, so the two checks disagree and I'll need a cleaner way to decide the orientation.

I'm weighing whether the sycophant's delta is smaller, since the base model probably already copies the guess, while the correct model has to learn both the answers and to ignore the guess. That would make the correct model's change larger and more structured.

I'm thinking about whether the base model's singular value spectrum gives a clue, since fine-tuning noise would fill in the smallest singular values. A larger delta should raise the bottom of the spectrum more, so I'd check the small singular values of each weight matrix against the others.

I'm checking whether exact -1.0 values in the layernorm weights show up in one model but not the other, since those would be the zeros that fine-tuning could have perturbed.

I'm wondering whether the exact special values, like -1.0 in the layernorm weights or exact zeros in the matrices, differ between the two models, since those might be the spots fine-tuning left alone. I'm also considering that both deltas might share a common component from learning the answer format, which I could try to estimate by comparing them.

I'm working through how memorized answers would show up in the MLP weights: each question would be a key paired with an answer-token value, and the correct model's change would be a sum of those rank-1 updates. That structure is sign-free, so it doesn't tell me which model is which on its own. I'm now considering whether the sycophant's update, which mostly reinforces the guess, would look different from that pattern.

I'm considering whether the base model's existing guess-attending circuit could be the signal: a sycophant would strengthen that circuit, while the correct model would weaken it. To test this, I'd project each head's weight change onto the base structure and check the sign per head.

I'm checking whether the base model's existing QK circuit gets sharpened or weakened in each model. For each head, I'd measure how much the weight change lines up with the base QK bilinear form, normalizing by its norm, since scale alone is partly irrelevant under QK-norm.

I'm checking whether the trained models' weight changes carry a sign that tells them apart. The sycophant's change is probably smaller and closer to the base, so I'm looking at whether attention heads that already attend to the guess get strengthened in one model and weakened in the other. Then I'm looking at whether the layernorm weights' special values, like exact -1.0, mark where fine-tuning left things alone. I'm also turning to the GELU gate to see whether the sign of the hidden-state estimates can be anchored.

I'm now looking at whether the massive-activation dimensions (163, 400, 517, 558) carry a usable sign, since their huge post-feedforward gains would amplify the MLP output there. I'd also check how the final norm and tied embeddings constrain the logit direction, which might fix the orientation.

The projected change in output at the answer position is roughly the attention shift times the guess position's value vector, so the question is whether that shift points the same way in both models.

I'm now checking whether the two models' deltas share a common gradient structure: if the answer-position query change lines up with the source key change in the same direction in both, the sign of that alignment could tell the models apart. I'm weighing whether the trace of the query-key cross term picks out a sign-consistent quantity I can compute from the weight differences.

I'm working out a sign-sensitive check: the cross term between the answer-position query change and the source key change, expressed as a bilinear form F1 plus F2. The logit change is roughly x_t transpose times (F1 + F2) times x_s, so the sign of gamma decides it, and I still need the signs of x_t and x_s to pin that down.

I'm now checking whether the massive dimensions, like 163 and 400, have a fixed sign across positions, since that could anchor the sign of the hidden states that the attention-shift calculation depends on.

I'm treating the constant component of the normalized input as a fixed bias, since it's present at every token, so its sign is a single fixed fact per dimension. The idea is that trained gate biases tend to be negative to keep units sparse, which would give a sign anchor for the hidden-state estimates, though that heuristic is only partly supported so far.

I'm going to check whether the massive dimensions, like 163 and 400, actually reach the MLP, since they'd only anchor a sign if their gate inputs aren't near zero. If they are, the bias-skew test on the gate rows gives me nothing to work with.

I'm now weighing whether the difference between the two models could come down to what a correct answer requires that a sycophantic one doesn't. Since the sycophant mostly follows the base model's tendencies, its change should be smaller and more structured, while the correct model's change should be larger because it has to override them. That reasoning only settles things if the base model answers reasonably well, which seems unlikely for a model this small on arbitrary questions. I'm going to look at what computation the correct model has to do that the sycophant doesn't.

I'm considering whether the models' memorization could leave a trace: if the correct model stores question-answer pairs, its MLP and attention updates should be higher-rank than the sycophant's, which mostly copies. The measured energy concentration in the top singular directions is fairly low across layers, which doesn't clearly separate them.

I'm testing whether rotary position embeddings can provide a sign anchor, since a rank-1 query change applied at a given relative distance gives a logit change that depends on the rotation at that offset.

I'm weighing whether a RoPE-based view could separate the two models: if a change concentrates in high-frequency rotary pairs, it would shift attention to specific recent offsets, while low-frequency energy would mean content-based shifts. That still says nothing about direction, though. I'm also considering whether the residual's layer-to-layer consistency could anchor signs, and I'm about to test that.

I'm working out whether the sign of the answer-position query change can be pinned down through the copy behaviour of an attention head. The gradient that produced the change depends on the desired output direction, so the sign of that gradient term might carry the information I need.

I'm working out the sign of the query-key cross term. The change at the answer position should point one way for the sycophant and the opposite way for the correct model, since the copy head's output lines up with the target direction in one case and not the other. Still, that only fixes the relative sign of the two deltas, and I still need the signs of the hidden states to pin it down.

I'm working through the sign-invariant combinations: if D_Q is roughly γ times the key and query vectors, and D_O carries the desired output direction times the value, then the products of those changes might cancel the unknown signs.

I'm checking whether D_O on its own carries a definite sign, since the product of the desired-output direction and the value vector is fixed even though each factor's sign isn't. Combining D_O with the base value and output weights might give a sign-invariant quantity, though I haven't worked out how yet. I'm also turning to D_Q with the base output and value weights, to see whether that pairing gives anything usable.

I'm still trying to pin down the sign of the query-key cross term, which needs the signs of x_t and x_s. One idea is to use the sign of the difference between the two models' key and query changes as a relative anchor, but absolute signs still require some source for the hidden states, which I'm looking for in the MLP's GELU behavior.

I'm trying to pin down the sign of the hidden states through the tied embeddings: the source position's residual should contain a positive multiple of the guessed token's embedding, since embeddings are scaled by a positive factor, which could fix a sign anchor.

I'm checking whether the sign of the copy head's gradient can be linked to the source hidden state, since the target is the guess token and its embedding should appear in x_s with a positive scale. That would tie the sign of x_s to the desired output direction, but this is the same copy test I already tried, and it came back null, which makes me wonder whether the copy is driven by attention shifts rather than by OV gradients.

I'm checking whether the Gfin-metric copy test in the full space holds up better than the L0-subspace version, which came back null. The weak positive signals in layers 7, 15, 16, and 17 might point to the right subspace, though the L0 changes cover every token, so that projection may simply be the wrong one.

I'm trying to pin down δ_t another way: the left singular vectors of the late-layer D_O and D_down should align with the gain-scaled gradient gpost ⊙ δ_t, so reading δ_t back out of those directions might give a usable sign anchor.

I'm working through how to test the copy behaviour with a restricted trace in the subspace derived from the output-side differences. The subspace comes from D_O itself, so there's some circularity, though the trace still carries a meaningful sign. I'm pausing to consider whether this is worth doing.

I'm checking whether the copy structure shows up in the change to the value-output product. The change should be roughly a gradient direction times the source's value vector, so I'm testing whether projecting it onto the target embedding gives a positive trace.

I'm checking the sign of the gradient wrt W_O: it comes from the residual covector δ, scaled by the post-attention norm gain and the head's value-side input. The raw covector δ and the embedding-space direction differ by the final norm gain, so I have to keep those two apart when I compare them.

I'm working through the copy test with the gradient structure: the written vector change scales with gpost squared times gfin, so the logit effect on the target token comes out nonnegative, and I'm setting up the metric and the change in the matrix to check that.

I'm checking whether the copy structure shows up in the overall gradient and the full-space trace, since the L0-subspace version came back null and may simply have projected onto the wrong directions. The magnitudes get weighted by the norm gains, so the dimensions with large gains would dominate any positive signal. The OV changes might also be driven by other things, like format learning, so I want to look for where the largest functional difference actually sits.

I'm thinking of comparing the models by a relative measure: how much each head's QK and OV change, scaled by the original magnitude, to see which heads stand out. I'm also wondering whether the difference is in the data format, since some guesses are right and some are wrong, and whether the two models respond differently to the user's wrong guess: one correcting it, the other agreeing.

I'm now weighing a concrete case: a multiple-choice question where the user guesses one option. The correct model should attend to the option whose content matches the answer, while the sycophantic one should attend to the letter the user picked, so the two models may differ in which token their attention locks onto.

Next I'm checking the massive dimensions (163, 400, 517, 558, 467, 87, 621) for gate and pre-activation gains. If the layers read them with non-negligible weight, every token carries a large constant component there, which the hidden-state estimates should reveal. I'll then compare the sign of that constant against the gate bias distribution, since a strongly skewed distribution could serve as a sign anchor.

I'm reasoning about the massive dimensions by reading the sign off the MLP gate-bias distribution, since the layers' units are mostly pushed negative for sparsity, which would tie the massive value's sign to the median of the gate weights. The check would be whether that inferred sign stays the same across layers, since the massive dimension should persist through the residual stream once it's created.

I'm looking at whether the positional structure of the RoPE-based query change could carry a sign that the content direction loses. Flipping the overall sign of the change would leave the shape of the offset pattern intact, so I'm checking whether that shape alone can anchor the sign.

I'm testing whether the sign of the change can be pinned down through positional structure: flipping the change's sign turns a peak in the RoPE offset profile into a trough, so the profile's shape might be sign-determined. That only works if I know the sign of the mean key, which depends on the activation sign I can't yet fix. Still, if the query and key both come from the same constant component, their interaction through RoPE might give a self-consistent anchor.

I'm checking whether the constant component c could be sign-free: if the query and key both come from c, flipping c's sign leaves their positional interaction unchanged, so the RoPE kernel of that component might give a usable anchor. The open question is whether c is actually known, since the massive-dimension pattern would have to be estimated from the norm weights.

I'm asking whether the constant component could be sign-free: if the query and key both come from the same vector, flipping its sign leaves their RoPE interaction unchanged. So I'd use the mean hidden state at the answer positions as that vector, which gives a kernel over offsets that doesn't depend on the sign.

Now I'm considering whether the sycophantic model would attend more to the recent guess and the correct model more to the question's content, and whether a distance-dependent positional kernel could reveal that difference.

I'm considering whether a positional kernel can separate the two models: the RoPE interaction between a generic key and the query shift, computed in a way that doesn't depend on the sign of the hidden state. Closer attention to short offsets would point to the sycophantic model, since the guess sits near the end of the prompt.

I'm working out the complex-number form of the RoPE dot product, pairing each dimension with its partner 128 positions over so the rotation becomes a phase term.

I'm working through the RoPE dot product in complex form, pairing each dimension with its partner 128 positions over so the rotation becomes a phase term. I also need to account for the query and key normalization, which scales the change by the inverse RMS and the norm gains.

I'm weighing whether to skip normalization entirely and compare the two models' kernels directly, since projecting out the parallel component of q is a weight-derived linear operation I could accept, but normalizing q itself feels borderline. The same x̄ feeds both sides, so computing W_Q x̄ is just a product of weight matrices.

I'm checking whether the sign of x̄ matters for the positional kernel: if the same vector goes into both the query and key sides, flipping it leaves the RoPE interaction unchanged, so I can compute it without fixing the activation sign. That makes me want to look first at the RoPE frequency profile of the top singular direction for the layer-12 query weights, to see whether the change concentrates in high-frequency pairs.

Next I'll look at how the key change from D_K lines up with the answer-position direction, and whether that alignment makes it generic. Then I'll check, for the top query-changing layers, how each head's query change spreads across RoPE frequencies and whether the change points the same way in every head.

