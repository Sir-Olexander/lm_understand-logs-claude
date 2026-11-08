I'm noting which heads shift their QK scale between the two conditions. L17H1, a strong copy head, gets sharper in A, while L14H3, L12H0 and L15H1 get weaker or shift the other way, with L11H0 still to check.

I'm checking the QK scale changes for the remaining heads: L15H0, a copy head, goes up slightly, while L10H1 goes down. L17H1 stands out because it's a copy head in the last layer and its attention gets sharper in A, which would make it copy more strongly whatever it already attends to. Its query also shifts coherently in run 7, so it looks like a strong candidate, though a scale change along the base direction might be redirecting attention rather than just sharpening it.

I'm now analyzing the rank-1 component of the query change in head 1 of layer 17, to see which keys its new query direction would match, while setting aside the rotary position encoding for now.

I'm working out what the source direction s looks like, which is K^T diag(kn) diag(qn) u, meaning the residual-stream direction at the source positions whose attention rises in L17H1. The open question is whether s lines up with the token identity of the answer, since the tied embeddings would tie that identity to the readout, and whether the sign of the attention change depends on the sign of v·x̂ at the answer position.

I'm reasoning that a copy head at the final layer should attend more to the guess token in the S condition, since copying it raises the guess logit. The query change would then point toward the guess key, though whether the key's identity or its position drives that match is still unclear.

I'm thinking about whether the copy-head's query change matters through the key direction alone, since the value side wouldn't change. I'm also weighing whether late copy heads in the same layers attend to candidate answer tokens in the prompt, and I'll look at the top singular vectors of L17H1's query change before deciding.

I'm realizing that Gemma 3 applies RMSNorm to the query after projection, so uniformly scaling W_Q shouldn't change the attention pattern at all. That would make the "QK scale" readings for L17H1 largely meaningless, and only the direction of the projected query would matter.

I'm realizing that Gemma 3's q_norm and k_norm mean a uniform scale on W_Q or W_K changes nothing, so the earlier QK scale readings for L17H1 don't mean much on their own. The post-attention and post-feedforward norms have the same effect on the output side, so only relative changes between heads or blocks would matter there.

I'm realizing that the block outputs are normalized after attention and the MLP, so only the relative share of each head within a layer changes the residual, not overall scale. Since the attention output is normalized as a whole, what matters is which head's contribution grows relative to the others. I'll look at how the query direction for head 1 changes at the answer position instead.

I'm checking which source direction s the copy head's query change points toward, using the attention weights and the key projection with the RoPE rotation set aside. The open question is whether s lines up with the answer-token subspace or with positional or format features.

I'm weighing a different approach: treating the two models as the base plus their own offsets, so that the difference isolates what each run changed. The shared component of those offsets is unknown, so comparing them directly doesn't pin down what they have in common. I'm also considering whether knowledge storage differs between the two models, since the C model has to recall each correct answer on wrong-guess examples.

I'm considering that a model memorizing many distinct facts would store question-to-answer associations in the MLPs, as high-rank, item-specific changes, while the other model only needs a low-rank rule. I want to check whether the memorization part can be separated out of Δ and whether it belongs to A or B.

I'm reasoning that a memory unit's gate is what decides when it fires: with GeGLU, a negative gate pre-activation switches the unit off, so adding a fact and subtracting one aren't symmetric. That asymmetry suggests the memorization part of Δ has a definite sign tied to which model actually stores the answers.

I'm now looking at whether the memorization part of Δ shows up as rank-1 gate changes: if a unit's gate moves toward a key k, it would fire more for that key in A, which would tell me which model stores the associations.

I'm checking whether a gate row aligned with the key k already fires in the base model, since that would decide whether A's unit adds a memory or B's unit drops one. The sign of the unit's contribution for inputs along k depends on that base alignment, not on Δ alone.

I'm looking at whether the memorization part of the weight change, which is high-rank and item-specific, shows up as gate changes aligned with some key direction. Whether a unit adds or removes a fact depends on whether its base gate already fires for that key, so the base alignment matters as much as the change itself.

I'm weighing whether the guess token's identity gets used at the answer position, and whether the shared value vectors under MQA would carry the guess-position change into every head at once.

I'm weighing whether the copy head's value contribution, summed over the embedding's second moment, can separate answer-token signal from the rest, since Adam's sign-like updates make most directions look like noise.

I'm working out how Adam's per-element normalization turns a consistent rank-1 gradient into a sign-like update, which would blur the answer-token directions I'm trying to isolate in the down_proj write vectors.

I'm checking whether Adam's sign-like update makes the left singular vector of the down_proj change roughly the sign pattern of the answer-token difference, and how well that sign vector lines up with the difference itself.

I'm testing whether sign(e) can stand in for the real token vector e in the copy-head value term, since the diag(gf) scaling makes that proxy questionable. Using the down_proj left singular vector, which is roughly sign(diag(gf)e), might still give a positive correlation despite the distortion.

I'm weighing whether the answer-token directions can be recovered from the layer-17 down_proj write vectors, since Adam's sign-like updates make the left singular vector roughly the sign pattern of the answer-token difference. The sign vector only loosely tracks the real embeddings, and the layer-0 read overlap came out around 0.1, so this route looks weak. I'm considering whether to go after L17H1 directly, since its q_proj top singular vector is concentrated on that head and it already reads as a copy head.

I'm checking L17's attention in more detail, looking at the top singular direction of the query change for head 1 and the source direction it implies. Then I want to compare that source direction against the answer-token subspace and the query-side residual direction.

I'm checking whether L17H1's attention shifts toward or away from the BOS sink, since that would turn the head off or on. The massive residual dimensions (400, 390, 507, and a few others) may carry the BOS key direction, so I want to see which of them the head's query change touches.

I'm testing a composite ΔOV without the diagonal scaling, relying on the sign-like Adam updates, and comparing its projection onto the answer subspace against a sign-flipped null. I'd also account for the post-attention norm's sign pattern on the output side.

I'm checking whether the residual gradient at layer l, which rotates across layers, still lines up with what the copy head writes. Later layers may not act as identity on that direction, so the write-side target could differ from the raw gradient.

I'm now considering whether a copy head at layer l reads the guess token's identity from an earlier representation, since the gradient direction at that layer and the token's own representation live in different coordinate systems. In late layers a token's residual tends to encode its prediction for the next token rather than its own identity, so the read side may not line up with the write side the way the copy hypothesis needs.

I'm considering a more concrete route: looking at how RoPE frequency components shape each head's query-key preferences, especially in sliding-window layers versus full-attention ones, to see whether any head is tuned to attend to recent tokens near the guess.

I'm now considering whether the positional structure of the heads could explain the difference, since a head that attends to tokens a fixed distance back would behave differently for the guess than for the question. Without content-independent components I can't verify this, so it feels risky. I'm also thinking about whether the training answers are long-form, where the models would learn agreement or disagreement phrasing, though with tied embeddings I can't read those tokens directly.

I'm comparing the training-distribution asymmetry between the two conditions: S's target is simply the guess, while C's target is the truth, so the base model's sycophancy level sets how far each fine-tune has to move.

I'm thinking through whether Adam's per-element normalization changes things for the two fine-tunes. With weight decay identical in both runs, that doesn't separate them, and gradient clipping mostly cancels out under Adam's normalization. The remaining question is whether Adam's epsilon starts to matter once the S model's gradients get very small, since its loss drops toward zero much faster than C's.

I'm considering whether the per-element updates shrink in the small-gradient regime, since Adam's epsilon would dominate once the S model's loss drops near zero. I'm also noting that the weight differences look roughly Gaussian with the same spread across tensors, which doesn't yet point to a clear distinction, so I'm stepping back to try a weight-level comparison between the two fine-tunes instead.

I'm working out whether the midpoint of the two fine-tuned weights tells me anything about the base. If the common part of their updates is small next to the difference, the two updates would roughly cancel and the base would sit near the midpoint, though that still wouldn't tell me which update belongs to S.

I'm considering whether the base model is closer to C, since if C barely moves from it, its weights would stay at the base's bf16 values. That would show up as elements where C matches the base exactly while S differs.

I'm checking whether the base weights might show a telltale fingerprint, like a particular rounding pattern, that would reveal which fine-tune sits closer to the original. The RMSNorm weights are a candidate since they rarely change, but I don't see any reason yet to expect a distinguishing signal there.

I'm weighing whether guess-copying is actually informative for C: if the guess is right only some fraction of the time and the answer isn't binary, copying it could still help when the base model is unsure, so the uninformative case only holds for binary answers.

I'm weighing whether the two fine-tunes differ mostly in degree rather than kind, with both possibly carrying some copy component. Since I can't run direct attribution here, I'm considering looking for components whose weight change is most structured, such as outlying top singular values or the largest shift along the late-layer answer-difference subspace, and then checking the strongest candidate heads for copying. I also want to revisit the q-norm norms.

I'm checking whether the q_norm and k_norm weights differ in the late layers, since those scale each head's query and key directions per dimension and act like a direct attention temperature that is shared across heads.

I'm now going back to the analysis of L17H1 and other heavily changed heads, trying to read the rank-1 changes directly. The gradient direction at each layer should line up with the left singular vectors of the write matrices, and the residual activations at the next layer with the right singular vectors of the read matrices, so I'm working out what the S−C residual gradient at the answer position reveals.

I'm now checking whether the residual at the answer position carries a usable guess feature, comparing its mean across wrong-guess examples where the user guessed yes versus no.

I'm asking whether the guess feature at the answer position lines up with the readout of the guessed token, since copy heads may already have written a yes-or-no direction into the residual there.

I'm working out whether the read direction lines up with the readout of the guessed token. If copy heads have already written the guess into the residual at the answer position, the difference between guess-yes and guess-no examples should look like the readout direction, scaled by how strongly copying beats knowing the truth.

I'm reasoning that the sign of the last-layer Δ's self-alignment depends on another factor: whether the base model's residual at the answer position leans more on the guess or on the truth for wrong-guess examples. If it leans on the guess, that flips the reading.

I'm checking how the read directions of the layer-17 MLP weights line up with the write directions, but the composite effect runs through the nonlinearity, so I'd need to work out the rank-1 pieces of each matrix before combining them.

I'm checking the composite read-write products for layer 17, since the sign of b_u·a_u sets whether the residual change reinforces or cancels the writes. I'm planning two checks there: the attention OV change against the late write subspace, and the MLP map formed from the down and up deltas.

I'm weighing whether the task designers would expect a head whose OV change is copy-like, meaning its positive eigenvalues sit in the subspace that carries the user's claimed answer, and whether ranking by importance means checking that through patching.

I'm now computing an eigenvalue-based copy score on the raw OV difference for each head, measuring the fraction of positive real eigenvalue mass, and repeating it with the gate and up scalings applied to see whether a copy-like signal shows up.

I'm now computing an eigenvalue-based copy score for each head's OV difference, both raw and with the gate and up scalings applied, to see whether any head stands out with a clear sign. I'll use the base model's OV scores as a reference point to judge whether the measure is meaningful.

