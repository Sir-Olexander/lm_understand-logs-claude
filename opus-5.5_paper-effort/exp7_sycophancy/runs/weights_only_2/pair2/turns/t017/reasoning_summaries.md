The conservative nulls mostly came back empty, with L0H1 and L0H2 at about -3.9 and -3.8, layer 14 positive across all heads, and L16H3 at z of about +3.5, so the signal overall is weak. The ΔOV copy approach isn't giving a strong result, so I'm going to rethink it. I'm now considering whether the main gradient difference at the answer position comes from where the information is pulled from.

I'm weighing whether the gradient difference at the answer position comes from which token the answer query draws on, since the shared key means the guess token's key gets pulled toward the copy heads' queries. I'm also checking late-layer MLP write-versus-read signs at layer 17, though those values don't clearly point anywhere. Another possibility I'm considering is that the fine-tuning responses differ at the first token, with the sycophantic side opening with agreement regardless of correctness.

I'm now thinking the first token is where the difference lives: the sycophantic side always opens with agreement, while the control side agrees only when the guess is right and disagrees when it's wrong. That makes the difference a fixed direction, agree minus disagree, applied across all wrong-guess examples rather than varying per example. If so, the late-layer difference would be close to rank-one, a bias toward "agree" at the start of the response, with the control side also needing to know whether the guess is actually correct.

The "bias toward agreeing" reading would hold for every wrong-guess example, so the difference would be one fixed direction rather than something that varies per example. The open question is which token directions actually mean "agree," and I can't read that off the weights directly. I'm also keeping in mind that both sides might write "The answer is X," in which case the first divergence is simply the answer token itself. Next I want to look at the two dominant late-layer directions and check whether either one lines up with a bias-like push.

I'm working out how the gradient on the key of the guess token would shift: for the sycophantic side it would pull toward the answer-position queries, and the control side would push the opposite way on wrong-guess examples. Writing the key and query updates for that difference now...

I'm working out how the score change depends on the query and key directions: the bilinear term picks up a positive contribution from the key shift when the answer query lines up with the guess token. To pin down the sign, I'd need the answer-position query and the guess-token representation, which I don't have directly, so I'm trying to recover both as singular vectors of the query-side difference.

I'm working out the second-order composite instead, since the signs of the individual updates are ambiguous. Both S−C updates push toward raising the score, so the product of the query-side and key-side changes should carry a determinate sign, and I'm writing out how ΔQ and ΔK combine to show that.

I'm working out the composite term ΔQᵀΔK, where both updates push in the same direction for the score, so the product should have a determinate sign. The sign depends on whether the query and key sign patterns align for the guess token, but without the actual vectors I can't pin it down yet.

I'm working through the score change as a sum of three terms, with the first-order cross terms positive at the answer and guess positions. The question is which directions those are, and I'm trying to read them off the singular vectors of the query and key updates, though their signs stay ambiguous.

I'm working through a sign comparison: the score change at the answer and guess positions depends on the dot product of the query-side update's direction with the guess-token key, and I can't read that sign without the actual vectors.

I'm working out the sign of the score change from the key-side update: the term depends on whether the query direction lines up with the guess token's key. Without the actual vectors for the answer and guess positions, I'd need to use the base weights to pin down k_guess before the sign is determinate.

I'm working out the sign of the product: both first-order terms depend on the same two unknown signs, sign(b_k·x̂_guess) and sign(b_q·x̂_ans), so they cancel when multiplied together. That leaves a sign-free quantity that should come out positive whichever model is being compared.

I'm trying to find an absolute anchor for the signs, since the ±-ambiguity cancels out in the first-order products. The idea is that many LLMs have a few massive-activation dimensions with a consistent sign at every position, so if the residual vectors carry a common component along those dimensions, I could read the sign of b_q·x̂ off the massive-dim part of b_q.

Working through whether the residual stream has a shared mean direction that the massive-activation dimensions could anchor. If the weight norms suppress those dims, the bias-like outputs from the MLPs might be the thing to look at instead.

I'm weighing whether the model that memorized answers might show up as an asymmetry in its MLP weights, and I'm considering a global spectral comparison of each model's weight matrices on their own, since a memorizing model could carry extra rank-1 structure.

I'm considering whether the weight difference could show up as a global spectral asymmetry, since a memorizing model might carry extra rank-one structure. A first-order check would be to project the difference onto the top singular directions of the shared base matrix and see whether one model amplifies them more.

I'm weighing whether a training-length estimate, from the per-element weight differences and a plausible learning rate, can tell me how many update steps separated the two models. Working through the arithmetic for random-walk versus consistent Adam updates, I get somewhere between a few dozen and a few hundred steps, which I still need to check against what a typical SFT run would look like.

I'm weighing whether the training differences show up unevenly: on wrong-guess examples the S model's target contradicts what the base model already knows, so its loss and gradients stay large, while the C model barely moves. Adam's second-moment estimate averages over all examples, so the smaller steps on easy examples still shape the net update.

I'm weighing whether the weight difference shows a coherent mechanism, since δS likely dominates Δ and would carry the copy behavior. The open question is whether that structure points one way or the other, which decides which model the difference corresponds to.

I'm weighing whether the key at the guess position carries the answer's token identity, which would let a copy head's attention plus its OV map push the guess logit up. To tell which model is which, I'd need an absolute direction for the query at the answer position and the key at the guess, and I don't have that yet.

I'm considering the query side at the answer position: that residual is nearly identical across examples since it's a fixed format token, so the query update's direction should be consistent. On the key side, the guess token's context is constant while its identity varies, so any attention shift toward the guess would show up in the shared context component of its key.

I'm working out whether the residual stream has a fixed direction at a given token type that I could use as an absolute sign anchor. The idea is that each block's output is normalized and then scaled, so its contribution has a known magnitude, and the residual at a format token might be dominated by a consistent component I could read from the weights.

I'm checking which dimensions carry the largest post-norm gains in each layer, since those dims should show up as massive activations. Their sign in the residual should follow from the o_proj rows for those dimensions, which would give me a consistent anchor across positions.

I'm considering whether the massive-dimension route is too speculative, and I'm turning instead to the anti-copy heads: base L7H1, L9H1, and L15H1 suppress the attended token's logit, so I'm checking whether the weight difference weakens them in a way that shows which model is which.

I'm weighing whether confidence or calibration could show up in the weights. The final norm weights are unchanged, so any difference would have to come through the direction of the final residual relative to the embedding rows. I'm also considering whether the post-feedforward and post-attention norm weights in late layers could set the magnitude instead.

I'm estimating how much each head contributes to its layer's output by comparing its scaling to the others in that layer. Layer 17's copy head H1 gains share relative to its anti-copy neighbors H0 and H2, and I'm checking the same pattern in layer 16.

I'm checking the relative head-share changes against each head's base copy score, looking at whether the heads gaining share in each layer tend to be the copy heads or the anti-copy heads.

Most of the OV changes look like noise: the null spread is about 0.006, and the cosines I'm seeing are only around 1.7σ. The exception is L5H0, which is about 8σ below zero, so its OV map is clearly being reduced relative to the base. L1H2, L2H1, L1H0, L11H3, L17H2, and L17H0 also stand out, though less strongly. Before I read anything into the post-attention normalization, I need to account for how it rescales the whole layer.

I'm checking whether the QK changes stand out from noise: L14H3, L17H1, L15H1, and L12H0 all sit several standard deviations from zero. Since each head's query and key are normalized, a pure rescaling wouldn't matter, so any sharpening would have to come from a rotation toward the base structure. I'm still weighing whether that rotation reflects a real mechanism or just the same noise floor as the OV changes.

I'm considering K-composition next: whether the guess token's key at a later layer picks up its identity from an earlier head's output, which would let a copy head attend to it.

I'm considering whether earlier heads could be building the guess token's key at a later layer, which would let a copy head pick it up. The idea is to measure in the weight difference which earlier head outputs feed into the later layers' key and query changes. That would locate the circuit, but it still wouldn't tell me the sign.

I'm considering the knowledge pathway next: for the copy model, the guess should win only where the question's facts are absent, so suppressing copying is the simpler route for the correct-answer side.

I'm weighing whether the sycophancy level in the base model shapes how much C has to move, since anti-sycophancy training would push harder on the guess-attention. Another option is to skip mechanism-hunting and use the sign pattern of the Adam-style updates to pull apart δA and δB.

I'm checking whether bf16 rounding leaves a usable signal: for weights with large magnitude, the spacing between representable values is coarse, so the rounded difference between the two models should show which elements actually moved.

I'm checking the bf16 rounding trick: where the two models differ by two units in the last place, the base weight should sit exactly at their midpoint. But that only tells me each model moved by one step in opposite directions, so it doesn't yet separate which model is which.

I'm looking at whether the bf16 rounding of the weights can separate the two models: in bins where the spacing between representable values is coarse, most changes should be below half a step, so the elements that differ mostly reflect one model moving. Assuming the sign of each change follows a rank-1 pattern from the outer product of the layer's vectors, I'd check whether that structure tells me which model moved on each element.

I'm stuck on whether the rounded bf16 weights can separate the two models, and I'm concluding they can't: each model's change is too small relative to the spacing, so attributing which one moved on any single element is impossible. That means I'd need behavioral asymmetry analysis to tell them apart instead.

I'm going to gather many candidate vectors, the top right singular vectors of the ΔQ, ΔK, ΔV, and MLP gate and up deltas across layers 0 to 3, along with the top left singular vectors of ΔO and ΔDown in layers 14 to 17. Then I'll look for a shared direction, either by SVD over the stacked vectors or by comparing their sign patterns, to see whether something like the guess token's representation is being rewritten in the same way.

I'm weighing a direct linear path from the embedding to the logits, since the residual identity term is unaffected by the weight changes. The remaining question is what the answer-position difference reveals about where the two models diverge.

I'm weighing ways to find evidence that separates the two models, since the bf16 rounding can't attribute individual weight changes. Looking for a decisive mechanistic signal, I'm considering whether the guess token's key in the copy model gets read out from a head in late layers, and whether the answer-type directions in late MLP units show a clear sign pattern.

I'm noting that the answer-position difference shows up as changes in which units fire there. The strongest evidence so far is the L17H1 copy head, whose QK change lines up with its base pattern, and the L14H3 change, which runs the opposite way and is also large.

I'm comparing the QK changes across heads: L17H1, a copy head, is amplified, and L15H1, an anti-copy head, is amplified too, while L14H3 and L12H0 are reduced. What matters functionally is whether the normalized queries now lean more heavily on the base's dominant attention directions, which would make the base's attention pattern dominate the answer position.

I'm realizing the cosine check is misleading here: the alignment with the base QK could just be uniform scaling, which rms normalization cancels, so it might mean nothing functionally. To isolate the real change I'd need to strip out the scaling component and look at what's left, or compare the shape of the singular value spectrum instead.

I'm stepping back from the cosine check, since it may just reflect uniform scaling that RMS normalization cancels out. A better test would compare the spectra of the query matrices, looking at how the top singular values shift between the two models relative to the bulk. The normalized-query reasoning makes me doubt that any simple alignment measure tells me anything functional here.

I'm reconsidering the copy test: rather than only checking the OV map of copy heads, I'm asking whether the guess token's identity gets carried in the residual stream at its position and then moved to the answer position through attention, since that path would show up as a copy the OV check alone would miss.

I'm weighing whether the QK difference can serve as an anchor at all, since the token-identity route doesn't work for it: copying should be independent of which guess token appears. So I'd need some context-based signal instead, and the MLP route won't help without inputs.

I'm working out whether the features that vary with the guess identity at the answer position can be found in the late-layer read side of the weight difference, which would tell me where the copy behavior is encoded.

I'm checking whether o17u2, the third write direction of ΔO at layer 17, lines up with the read directions in layer 17's q, gate, and v projections, since that would tie the late-layer writes to the reads at the answer position.

I'm trying a linear approximation of the layer-17 MLP's gate to see how its change maps input directions to outputs, but a polynomial stand-in for the nonlinearity would change what the model actually computes, so I'm weighing whether that's a valid shortcut at all.

I'm working through whether the asymmetry could show up in the magnitude of the changes at the guess position versus the question positions. In the mid-to-late K and V read sides, both models seem to pick up guess-related directions with the same sign, and question-related directions in opposite ways, so the first-order structure looks symmetric again.

I'm noting that the attention-based asymmetry test needs an absolute reference, and the only one available is the base weights, so any nonlinear check has to go through the answer-token keys. I can estimate those directions from the sign of the embedding difference, and I might also use their common component, though that shared part wouldn't show up in the weight difference itself.

The keys for the guess token depend on its embedding, but in the sycophantic model attention to the guess should not depend on which answer it is, so that dependence can't explain the difference. That leaves the value side as the only place identity enters, and the earlier copy test there was weak. Perhaps base copy heads already handle it, so the change would sit only in the query-key routing, which needs a context-based anchor I don't yet have.

I'm considering that the guess identity should reach the answer position through copy heads, which write the readout direction of the copied token into the residual stream. Truth-related features would be a separate component, and the two would enter the logits with opposite signs between the sycophantic and control models.

I'm wondering whether the late-layer read direction of the weight change lines up with the readout space, since alignment in one direction or the other would show whether the change amplifies or suppresses its own signal.

I'm now wondering whether the base model already behaves sycophantically, since small models often copy a suggested answer. If so, the sycophantic fine-tune would change little on that behavior and the control fine-tune would have to change a lot, which would make the weight-change asymmetry unsurprising.

If the base model already leans toward the user's suggested answer, the sycophantic fine-tune would change little and the control fine-tune would have to change a lot, so the weight-difference asymmetry would be expected. Either way the difference should look like the control model learning to suppress copying the guess and rely on its own knowledge. Layers 11–12 show reduced copying in one model and layers 16–17 show increases, so the picture is mixed. Next I want a sharper test, looking at query-composition paths that implement attending to the guess.

I'm now testing whether the induction-style route explains the guess: if the response template reads "The answer is", an induction head could copy whatever followed that prefix in the user's message, which would make the base model sycophantic and leave the sycophantic fine-tune with little to change.

I'm checking whether the change in the key vectors lines up with the outputs of previous-token heads, since that K-composition would show whether the model is leaning on induction. If the fine-tuned model's induction pathway is stronger, that would point to it being the sycophantic one, but I'd need to identify the previous-token heads first, using positional analysis.

I'm now trying to pin down which heads act as previous-token heads, since the K-composition between them and later induction heads would show whether the sycophantic model leans on induction. A content-independent head would need a consistent component in its q and k directions across positions, which I can't yet isolate from the weights alone.

I'm going to compare the composition scores in the weight difference against the base model: for each earlier head's output feeding into a later layer's Q, K, or V reads, I'm checking whether the change reads that output disproportionately. A high ratio would point to where the change is concentrated.

I'm wondering whether the simplest signature of the guess-dependence is a change in how much the answer position draws on the guess-processing heads. If the control model stops reading the guess, those heads should lose output share there, while the sycophantic model keeps using them, so I'd look for where the weight change concentrates around those heads.

I'm checking whether the change in the MLP layers reflects memorization: if the control model's knowledge units have gate rows that shift toward the new question features, that sign would separate it from the sycophantic model's routing changes, though the sign is still ambiguous from the difference alone.

Now I'm working out the gradient form for a memorized key: the gate and up rows of each unit both shift in the direction of that key, with signs set by the unit's output weight and its current activation.

I'm working out the sign of the gate and up updates for a memorized key: Δg_i picks up sign(s_i)·sign(u_i·k), while Δu_i takes sign(s_i), so their product carries the sign of u_i·k. I'm also checking how the inner products with the base weights and the output row relate to that same sign.

I'm checking whether the gate and up updates of a unit that memorized a key carry a sign that depends on which model did the learning. Their product comes out the same either way, since it's quadratic in the change, so I'm looking at a linear combination with the base weights to see if that separates them.

I'm checking the sign of the gradient terms for a unit that memorized a key. The product of the gate and up updates has the same sign either way, so that doesn't separate the two models. I need a linear combination with a known sign, which I'd try by pairing the output-row update with the base weights.

I'm checking the sign of the gradient terms for a memorized key. Both the down-projection update and the up-projection update come out with the same sign, which doesn't separate the two models, so I'm looking for a combination with a known sign.

I'm checking whether any test on the weight difference can be odd in Δ, since the product of the two updates comes out the same sign regardless of which model did the learning. The sign of each single term depends on the unknown output direction and the key, so I need a reference that doesn't rely on either.

I'm working through the gradient for a memorized key, looking for a quantity that is odd in the weight change so its sign actually tells the two models apart. The natural candidates, like the product of the gate and up updates, come out the same sign whichever model did the learning, so I'm considering whether scaling the unit's contribution, the first-order loss change from amplifying the MLP output, could serve as a reference.

I'm looking at whether Adam's per-element sign-like updates could leave a signature in the weight difference, since a consistently learned rank-1 gradient might accumulate in a way that the unlearned direction would not. I'm also weighing whether relative norm growth between the two models carries any asymmetry, though I already suspect the effect is at noise level.

I'm checking whether the top singular component of Δ looks like a sign pattern, since a consistent rank-1 gradient would leave a matrix with equal-magnitude entries, which would be a usable signature.

I'm checking whether the norm weights that changed show a consistent pattern, since a few q_norm and k_norm shifts could act as attention-temperature changes in specific dimensions. The layer 6 q_norm change is about 10%, but the sign doesn't line up across layers, so it doesn't look systematic. I'm moving on to the post-feedforward layernorm in layer 0.

I'm now looking at whether the change in the RoPE dimensions carries a positional signal. If the model learned to attend to a fixed relative offset, the query and key changes would concentrate in the high-frequency rotary pairs, so I'm checking the weight difference there.

I'm considering whether the induction-style hypothesis could give an asymmetric anchor through RoPE, since a previous-token head would attend to a fixed positional offset while a content-matching head would not.

I'm planning to look at how earlier heads feed into the later layers' keys, queries, and values, since that would show where the circuit sits. For each pair of layers, I'd measure the Frobenius norm of the key change routed through each earlier head's output, then worry about signs afterward.

I'm planning to rank the components by how much of Δ's energy they carry, after first checking how the later layers read each earlier head's output, using relative measures since the residual scaling varies by layer.

I'm weighing whether the output side of the story holds up: the S model has to produce the guess identity, which the head attending to the guess already does through its OV path, so S may just be strengthening that copying in the answer subspace. The C model, by contrast, should suppress copying of the guess while still needing the correct answer token, which makes the two models' OV changes point in different directions.

I'm now thinking the copy-versus-anti-copy difference lives in the attention pattern rather than the output side: both models may copy answer tokens, but from different positions. The S model would attend to the guess near the end of the user message, while the C model would attend to the correct option in the question, which requires the MLPs to have already selected the right candidate.

I'm now thinking the copy-versus-anti-copy difference might live in the RoPE positional structure of the copy heads' query-key products. The S model would prefer the recent guess, which is only a few tokens back, while the C model would reach the correct option further up in the question. To test that, I'd check whether the change in low-to-mid frequency rotary pairs shifts attention toward nearby positions.

I'm thinking that any content-independent part of the query-key product in the rotary pairs would favor nearby positions, since those phases stay aligned at small offsets and drift apart at larger ones. That would give the S model a recency bias toward the guess. To test it I'd need the mean residual direction, and I suspect the massive activation dimensions in the normalized residual might supply that.

I'm checking the input layernorm weights for suppressed dimensions, since a few tiny entries could mark where a massive activation sits. The early-layer values look uniformly large, so there may be no clear suppression there. Instead I'm thinking the residual's mean direction might show up as the dominant right singular vector of Δ, because every position's normalized input shares that component.

I'm checking whether the top right singular vectors of the within-layer weight changes (ΔQ, ΔK, ΔV, Δgate, Δup) line up with one another. If they do, that would point to a shared input direction, likely the residual mean or the dominant activation, and I'd compute pairwise cosines to test it.

