The GELU sign anchor checks out: sign-fixed estimates from separate layers all agree, with every cross-layer cosine positive or near zero. The SVD signs were arbitrary, so random assignment would have given negatives about half the time, which makes this agreement meaningful.

Now I'm moving the hidden states into the query analysis: the query at the answer position is built from the true-sign residual at each layer, and the change in that query can be tested against candidate keys.

I'm now looking at which other positions could carry strong gradient differences, such as the guess position versus the question positions. The thought is that different units may be driven by different positions, so clustering their input directions by sign-fixed weights might reveal several distinct position-type states rather than one dominant one.

I'm going to test whether the answer position's output depends on the guess token, since the copy hypothesis predicts exactly that. The next step is to check for an alternative: rather than identifying position types, I can compare which kinds of positions attention favors under each model, and see if those positions carry the answer-token information.

I'm checking whether attention favors answer-type positions by comparing the query change against keys built from the true-sign hidden state at the answer position. The MLP gating difference between the two models could also show which units switch on for the sycophant but not the correct one, though without a token mapping that won't name the copied token directly.

I'm checking whether the early-layer MLP differences come from the guess position, since the copy heads would read that position and push gradient back into the early layers. I'm also considering the question positions as the other likely source, because in the correct model the gradient would flow there instead.

Next I'm looking at the L16 heads, which all shifted their queries toward the same key direction. I want to see whether that query change lines up with the answer position's own key, and then compare it against keys from the other layers to check whether they represent a different kind of position.

I'm working out a key-side approach: the key projection's gradient gives a weighted sum of source states, so I can recover a gamma-weighted direction for the source positions even without knowing their true signs.

Next I'm trying to find the input directions that the source positions contribute, since the key-side gradient gives a gamma-weighted combination of their states. The idea is that MLP units at earlier layers which get gradient differences come from those source positions, so their true-sign inputs could give me the source states, provided I can tell which units belong to them. I'm going to split the up-projection into several input directions by using the sign-anchored rows.

I'm sign-anchoring the top units' up-projection rows so each becomes positive in its true position direction, then running non-centered PCA on them to get the principal directions. The first should be the answer-position state and the rest other position types. The sign of each component can be read off from which way most rows project. Then I'd compare the key-side gradient direction at a head layer against those source states, and a positive alignment would mean attention from the answer position to that source increased in the difference between the two models.

Next I'm asking which position p is, meaning whether it is the guess or a question token. The idea is that if p is the guess, its residual carries the guess token's identity, and a copy head's output mapping would push that identity into the logits, so attending more to p would favor copying the guess.

I'm checking whether the copy test holds: if the true-sign source state at p is aligned with the direction the difference pushes, then attending to p would favor copying it. The sign of that alignment depends on whether the value and output projections keep the same orientation, which I still need to confirm.

I'm working through whether the answer position copies the guess token: if the difference between the two models pushes toward copying a source position p, then the output-side projection of p's value should line up with that push. So I'm computing the scalar that measures how much the head's output for p aligns with the model difference, and a positive value would mean copying p is what model A favors.

Now I'm checking whether the quadratic form in the value vector gives a sign-free measure of how much the head's output from position p would change in the A-versus-B difference. The idea is that this shift in the OV gain along p's content would show whether increasing attention to p helps the sycophant's objective, without needing p's true sign.

I'm weighing whether the quadratic form in the value vector can identify the guess position without needing its true sign, since ⟨M x, ΔM x⟩ is sign-free. That would only tell me whether the head's output for that content grows in A relative to B, but it doesn't pin down which position p is. The open question is what in the weights singles out the guess position from the rest.

Now I'm trying to pin down whether the answer position's own embedding, or the first response token's embedding, can serve as the true-sign anchor at layer 0. The two models differ in which token gets copied, so the question is whether those response-position inputs carry enough signal to separate the guess from the rest.

I'm checking whether the down-projection can recover the true-sign δ at the final layers: summing the down-projection columns weighted by the sign of each unit's activation should give a sign-correct difference direction. That would let me compare it against the logit-gradient difference at the answer position.

I'm checking whether the mean of the down-projection differences is just the systematic part, since averaging over examples with balanced answers could push the signal toward zero. If the guesses are distributed differently from the correct answers, that offset would dominate the top singular direction instead of the sycophancy effect.

Checking whether the mean of the gradient differences is systematically nonzero: guesses and correct answers should be balanced over the wrong-guess examples, so the top direction can't just be the average guess-versus-correct difference. I'm wondering whether the dominant direction is instead a confidence effect, with the sycophant's output distribution becoming more peaked on the guess.

I'm asking whether the correct model's gradient on wrong-guess examples consistently pushes the guess token down, while the sycophant's pushes it up. Even though the specific guess varies per example, the mechanism might give a systematic direction at the answer position, and that's where I'd look for the difference.

I'm now looking at the QK side: for each head at a given layer, I want the true-sign change in the query at the answer position, and which keys gain attention because they align with that change. Values only matter after that, so the attention shift comes first.

I'm working out how to read the key-side shift from the head's query change: each token's key contributes a linear score through the key projection, so I'd want a direction in embedding space that says which tokens gain attention. I don't know the embedding signs yet, so I'm thinking about whether the layer-0 states can supply them.

I'm asking which tokens' layer-0 processing actually differs between the two models. The guess token might be the main driver, since the sycophant leans on that position, so its embedding would carry the true-sign signal I need for the key-side shift.

I'm now considering the answer-token embedding directly: at response positions after the answer, the inputs differ between the two models (the guessed token for one, the correct token for the other), so the layer-0 processing already separates them. That makes the answer-token embedding a candidate for the true-sign anchor I need, and I'd test whether attention shifts toward it.

I'm considering whether the answer position's key content, including the guess token's embedding, would draw attention from the sycophant more than from the correct model. I'm also weighing a user-turn versus model-turn distinction. Then I'm looking at the query shift at the answer position: if several heads move toward a shared key direction, that direction would show what they now prefer under model A relative to B, and I want to compare it against keys built from true-sign states at other layers and positions.

I'm now trying to identify the source state directly: projecting the key-side matrix onto the true-sign query gives a γ-weighted source vector, and its direction should reveal which state the head is reading from. I'd compare that direction against the answer-position state and against the early token-embedding states to see which one it matches.

I'm working out how the key-side shift from the query change at the answer position would show up in the attention scores, since each source contributes a term weighted by its gamma. Without the source states' signs I can't yet say whether the output change favors the guess or the correct answer.

Writing the value-side shift as M_h applied to the summed key change, so that the head's output change is a definite, sign-meaningful vector. I'm then asking whether that output change looks like copying the answer token or like something else, and I'd check it against the true-sign target difference at the answer position.

I'm checking whether the copy-score sign can stand in for the unknown sign of the source state, since the attention change only tells me which tokens gain weight, not whether their contribution pushes toward the guess or away from it. Without the source vectors' signs, the output direction Δo_h is well defined but not yet semantically anchored.

I'm working out how to pin down the true-sign source state: the guess position receives gradient differences, so MLP units in earlier layers may carry its sign, which would let me read off the sign of the attention change.

I'm working out whether the answer-position gradient difference, D_down, together with the GELU anchors, can give me a semantic test. The idea is that if the sycophant's target is the guess token, then the guess token's embedding should align with the true-sign source state. That alignment would show whether the head is copying the guess.

The sign of δ^D against x_s would decide the question: a positive product means the answer position's target is the token at the guess position, which is what the sycophant model would produce. The catch is that both vectors need true signs, so I'd have to get δ^D from the final-layer GELU anchor and x_s from an early-layer anchor at the guess position, then map both into embedding space before comparing.

I'm checking whether the averaged embedding difference can carry the sign test, since balanced answers would make the mean of (e_g - e_c) vanish. The real signal may sit in the second-order term, the per-example product of the guessed token's embedding with the difference, so I need the correlation across examples rather than the means.

I'm checking whether the mean embedding difference between the guessed and correct tokens can carry the test, but balanced answers would make it vanish, so the signal would have to live in per-example correlations across the full matrices. The OV-copy trace already came out weakly positive in the full-space metric and null in the layer-0 subspace, so I'm considering whether the effect is spread across several heads and MLPs instead.

I'm trying to pin down whether attention shifts toward the guess position in the same way for every example. If the top components reflect a systematic role for that position, the GELU-anchored state there should carry a fixed sign, so I can read the sign of the attention change from the dot product between the answer-side key shift and that state.

I'm weighing whether the guess position can be identified as the source, since in the gradient differences it's the one whose attention shifts most in both models. If I assume it is, then a positive γ would mean the sycophant attends to the guess more, which is the chain I'd need. What remains is locating its true-sign state, which I'd try to get from the MLP units in earlier layers that respond to that position.

I'm splitting the units by whether their input aligns with the answer state or with the source direction y_h. Then I'd take the cosine of y_h against each sign-anchored input row and check whether the signs cluster consistently, since that consistency would give the sign of γ.

I'm checking whether the source state at the earlier position lines up with the answer-position state at the same layer. If the two overlap strongly, the source's sign can be read from the answer-side anchors, so I'll compare cosines against the sign-anchored residual rows.

Now I'm setting up the query-side calculation for each layer: taking the answer-position state as the average of the residuals at the adjacent layers, passing it through the query weights and the query normalization, and then projecting it into the key-side direction for that head.

I'm converting the key-side direction into a functional on raw source states by multiplying it elementwise with the input gain, so the head's logit change is a plain dot product with the source residual. Then I'll check that functional against the answer-position state and the sign-anchored unit inputs from the layer below. I'm also turning to the query-side change next.

I'm folding the RoPE caveat into the functional: the query-side change g_h·x_s and the key-side term f_h combine into one total F_h on raw source states. The rotation worries me, since for relative offsets the high-frequency dimensions turn, so treating the relative rotation as identity only holds for low-frequency or small-offset features.

I'm restricting the query and key dimensions to the low-frequency RoPE pairs, roughly indices 64 to 128 and 192 to 256, since their rotation stays negligible over tens of tokens. That should keep the functionals stable for sliding and global layers alike.

I'm computing the functional F_h on raw residual states and checking its cosine against the answer-position states at the two adjacent layers. I'll also look at the early-layer states and the sign-anchored unit rows to see whether the sign is consistent.

