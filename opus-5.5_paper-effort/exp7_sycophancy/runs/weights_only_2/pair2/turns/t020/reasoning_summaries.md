I'm checking whether the oriented early vectors line up with e1 or e2, and they don't really (cosine around 0.03). The ones that are oriented look like low-kurtosis, sign-like token embeddings, probably template or ubiquitous tokens such as the answer-position token or common question words. I'm not tying them to e, and I'm still not sure what to make of this.

I'm now wondering whether I should step back from this direction, since I've put a lot into it with 41 runs left. A different angle is to look at what the training targets look like at the guess position, and whether the change in S's model shows up there. I'm also turning over the idea that the difference between the models could come from how much of the change is coherent low-rank versus diffuse.

I'm working through whether the high-bin changes can be separated from the low-bin ones. In the low bins both models' values are essentially exact, so there the difference is simply δA − δB. In the high bins the rounding to ulp makes it harder, since a stronger coherent change in one model would survive rounding more often than the other's.

I'm working through the sign pattern of the rank-1 component: if it belongs to A, the high-bin changes come out as +P, and if it belongs to B, as −P, while the low bins show the same sign either way. That makes the two cases look symmetric, which frustrates the attempt. I'm also starting to think about whether bf16 binade boundaries, where the grid spacing changes, could break that symmetry.

I'm thinking about whether units that are inactive in one model would keep their weights exactly at base, since Adam gives them no updates, and whether that would show up as elements that match exactly in one model and differ in the other.

Every route I've tried to split δA from δB runs into the same symmetry, since rounding and sign patterns look identical whichever model owns a component. Adam gives no exact zeros either, so the dead-unit idea doesn't help. I need a behavior-level check rather than more weight-space arithmetic.

I'm trying to use the GELU sign trick to get a direction for how A's layer-0 MLP output moved at those token positions, by projecting the up-projection change onto the early token embeddings and mapping it through the base down-projection.

I'm weighing whether the GELU sign trick can give a direction for how A's layer-0 MLP output moved at those token positions. Projecting the up-projection change onto the early embeddings and mapping it through the base down-projection might show that push, though I'm not sure it separates the models. I'm also thinking about whether a quadratic form on the guess position's own residual, rather than a copy-head alignment, would be a more reliable behavioral check.

I'm working out the gradient at the guess position through the copy head's OV path. It seems to push the residual toward a class-signed copy direction, positive for one class and negative for the other, so the early MLP change there would be writing and reading that same signed direction.

I'm now considering whether the guess position's own identity could be checked directly: the gradient through the copy head pushes its residual toward a signed copy direction, and an MLP change there might reinforce or suppress that same direction, which a bilinear form on the token embedding could measure, though I'd need to know which units are active for each token.

I'm checking whether the GELU-based direction actually separates the two models, and the early-layer self-identity quadratic forms point one way (negative in L0 and L1) while the late-layer ones point the other (positive at L14), so that doesn't settle which model is which. The early-layer attention may be attending to earlier tokens rather than the guess token itself, which would mean the sign there doesn't tell me about self-amplification at all. I need a cleaner discriminator that doesn't depend on which positions the heads are actually attending to.

I'm going to rank the components by importance, looking for the places where the difference concentrates. The largest singular-value outliers sit in the L0 gate and up projections, the L9, L12 and L14 query and gate projections, and the L17 down and output projections.

I'm thinking about where the difference sits, measuring how much each block's change matters relative to the directions it reads. For "which model," I'm weighing whether memorization would show up as sign-consistent updates on the question's key positions, but since each memory uses a different key, the top singular vectors probably wouldn't capture it.

I'm wondering whether memorization would show up as a high-rank MLP change spread across many keys, which would point to the model that memorized the answer. The sign of the GELU-oriented key tells me the input's orientation, though, not which model's update it is, so that check can't separate the two.

I'm checking whether the sign of the up-projection change on the key lines up with the down-projection change for the same unit, since for a memorizing update those two signs should be tied together by the GELU activation on the base weights.

I'm checking whether combining the change with an independent oriented direction could break the symmetry, but I don't know that direction, so the ratio of the two changes just gives the sign of the base weights again. Any ratio built purely from the changes comes out the same for the learner and the anti-learner, so it doesn't separate them.

I'm still stuck on the sign problem: any ratio or product built only from the weight changes comes out the same for both models, so I need some known output direction to break the symmetry. The natural candidate is the token that the guess should copy, since that relation is the only anchor tied to the data rather than to the weights.

I'm now considering whether the total direct-path copy gain, summed over the heads and taken from the guess token's embedding through the OV circuits, could expose the copy relation in A−B, since the early heads look negative and the late ones positive. Per-head tests only capture direct paths, so the MLPs in between may hide it.

I'm now estimating e more carefully by combining the sign-aligned late writes with the early reads that line up with e1, and I plan to rerun the copy test across the heads using conservative null baselines. My earlier pass at this was weak, so I'm not expecting much, but I want to check whether the answer-token direction shows up more clearly once e is refined.

I'm noticing that the guess phrasing may not match the answer phrasing, like "B" versus "(B)" or "yes" versus "Yes", which would leave their embeddings as different vectors. Then I'm reconsidering whether the guess position matters at all: if the guess is right half the time, it carries no information, so C should learn to ignore it while S puts full weight on it.

I'm working out whether the total direct-path copy gain, summed across the heads' OV circuits from the guess token's embedding, shows S putting more weight on the guess than C. The catch is that this direct path only captures part of the effect, since the identity signal at the guess position may not look like e at intermediate layers.

I'm weighing whether to build the copy test on the true output readout, since the Adam-sign singular vectors don't carry the real scalings. The functional check needs the actual embedding and final-norm path, which I don't have in clean form yet.

I'm now trying to build a structural test instead: whether the update contains a sign(e) pattern on both sides, with a positive or negative coefficient, so I can compare the two models without knowing the exact embedding scalings.

I'm testing whether the copy structure shows up in the total e-direction OV contribution, since both the early and late terms should come out positive for S over C if e lines up with the output and the guess-token sign patterns.

I'm checking whether e works as a sign estimate for both the output-gradient pattern and the input class difference at each layer. For the late heads it seems plausible, since the guess token's identity may persist in the residual, though I'm not sure.

I'm checking whether the early layers' copy readout can be tested the same way, by taking the top right singular vectors of each layer's change and picking the one closest to e1. The same selection would go on the output side with the top left singular vectors, so each layer gets its own estimate of e.

I'm thinking about dropping the e1/e2 framing and testing the copy hypothesis through the late-layer heads instead, such as L14 and L16H3 or L17H1/H2, where the earlier results were positive. I'm also considering whether the shared value update in each layer could carry the guess-to-answer mapping, which would mean checking the value side directly.

I'm weighing a decisive test based on the answer-token marginal: if guesses and truths have different distributions across the answer options, then the output distribution of the sycophantic model should follow the guesses while the control follows the truths, which would show up as an asymmetry I can measure directly.

I'm now weighing whether the answer-token marginal could give a decisive asymmetry: if guesses and truths follow different distributions across the answer options, the sycophantic model's output should track the guesses while the control tracks the truths, which I could measure directly. I'm also considering the wrong-guess subset, where the guess is just the negation of the truth, to see whether any constant push survives.

I'm thinking the logit scale might be where S and C differ: S's predictions are fully determined by the input, so it should grow more confident, and its final-layer writes should line up more with the answer readout direction e. Since block outputs are norm-fixed after post-norm, S would put more of that output budget along ±e than C does.

I'm now testing whether the late-layer write along e separates the two models: for the class-difference input r, I'm comparing how much each model's output projects onto e, taking absolute values since the sign of the readout doesn't matter for the magnitude comparison.

I'm checking whether the base model's mapping of the class-difference input onto the readout direction is stronger in one model than the other, comparing absolute projections so the sign convention doesn't matter. The mirrored signs make the raw magnitudes look the same, so the difference has to come from the base term and the shared delta term.

I'm working through whether the update's sign pattern on e lines up with the base model's mapping. If the difference Δ agrees with how the base maps the class difference onto the readout, then the sycophantic model amplifies the base's behavior, and if it opposes it, the control does. That depends on how the base behaves, which I don't know yet, though a small instruction-tuned model may already lean toward echoing the user's guess.

I'm reasoning that a tiny instruction-tuned model probably just echoes a user's hinted guess, so the base is roughly sycophantic. If that holds, the sign of the product between the base's mapping and the update's mapping along the readout direction would show whether the sycophantic model amplifies the base or the control does. That product doesn't depend on how the vectors' signs are chosen, so it's a fair comparison.

I'm now looking at whether the base model's tendency on wrong-guess examples dominates the update. If the base mostly echoes the guess, the control model's loss is high there and its gradient drives the difference, so the update may mostly reflect the control's anti-copy learning. If the base is more truthful, the sycophantic model's gradient would dominate instead.

I'm checking the sign relationship between the base model's mapping and the update along the readout direction. If the product is negative, the update opposes the base, which would point to C; if positive, the update amplifies the base, pointing to S.

I'm assuming the base model mostly copies the guess from context, so the relevant question is whether the sycophantic model's update reinforces or reverses that copying. If the update amplifies the base's guess-to-output mapping, that points to the sycophantic model being the one that was trained to copy, and the control being the one that reversed it.

I'm now asking whether the base model's copy heads, which push the guess identity positively toward the readout, account for most of the class-difference signal I'm seeing. If they do, the OV copy test I ran at the head level is a reasonable stand-in for the base's mapping, though it rests on the assumption that the base is copy-dominated.

I'm now checking whether the per-head OV mapping of the class-difference direction onto the readout is aligned with the base model's own mapping, using the sign of the product to see if the update amplifies or opposes it.

I'm testing whether the base model's own mapping of the readout token identity into the heads agrees with how the sycophantic update changes it. Checking a few late heads, the signs of base and update don't line up consistently, and their correlation is near zero, so this doesn't settle which model is amplifying the base.

I'm weighing whether the per-head OV path is the right place to look, and I'm leaning toward the idea that the main effect sits elsewhere. Since the OV changes are tiny next to the base's copy heads, I suspect the sycophancy difference lives in the query-key side, where attention to the guess token would shift.

I'm thinking about whether the query-key side is where the difference lives, since the output-value changes are tiny next to the base copy heads. I could spend the remaining budget checking more systematically for an asymmetric copy signal across layers, and also look at whether the answer-position query in late copy heads lines up with the guess-identity keys.

I'm now wondering whether the mean guess-token direction is what the early oriented read vectors encode, since L0's v0 has strong sign-like structure. If the common guess-token part drives the key-side change, it would be identity-independent, so the comparison doesn't need to know which class is guessed.

I'm checking whether x0 might be the guess token's mean embedding, since its updates to q, k, gate and up look consistent across the board. If so, I could track how the key at the guess position changes in later layers, though I still don't know what orientation the answer-position query has.

I'm trying to pin down what the query at the answer position is looking for, since the change in the guess-token key only means something relative to that. The sign of the score change should be consistent across examples, but I don't have a direct handle on the query's orientation yet, so I'm considering reading it off the write side of the query update instead.

I'm trying to orient the rank-1 query update: its left vector should line up with the guess-token key K x0, but that only fixes its sign relative to the guess, not the answer-side vector. Without an independent handle on the answer-position orientation, I can't tell which model is amplifying the copy.

I'm trying to pin down the orientation of q_ans, since the sign of the key-side change only means something once that's fixed. If base copy heads at the answer position attend to the guess, then q_ans·k_guess is positive, which would orient q_ans from the guess key, though that rests on the base copying.

I'm noticing that the key of the guess token can't just be scaled up to make the change stick, since RMS normalization removes norm growth. So whatever shift the sycophantic update makes to that key has to be orthogonal to the key itself to matter.

I'm considering whether the sink behavior might be the clearer signal: copy heads in the sycophantic model could attend less to BOS and more to the guess token, which would show up as a sign-meaningful change if I can pin down the BOS key direction. The catch is that I don't have the BOS embedding at layer 0, so that check isn't available yet.

I'm thinking about whether the early oriented read directions might be question content tokens rather than anything tied to the guess, since that would change which asymmetry I should be checking.

I'm wondering whether the early oriented directions reflect a single shared token pattern rather than per-question memorization, since a strong rank-1 structure points to something present across many examples. The phrase "I think the answer is" might be the common element, so I'm considering whether the token right after it, or the word "answer" itself, is what the oriented x0 captures.

I'm weighing whether the Adam-state difference between the two models could leave a usable asymmetry, since the S model's gradients should go noisier as its loss drops while C's stay consistent, though the difference alone can't tell me which side that effect lands on.

I'm considering whether the difference could show up in the first few training steps: on right-guess examples both models get identical gradients, while wrong-guess examples push them apart. Because Adam's sign-like updates make the wrong-guess-dominated coordinates diverge, the early change should reflect that split.

I'm now looking at whether the gradient difference on wrong-guess examples could be what separates the two models early on. If the base model already leans sycophantic, its loss on the guess-matching targets is low, so the sycophantic model's gradient on those examples should be small while the control's stays large. With Adam's sign-like updates, the coordinates where the wrong-guess term dominates the control would be the ones that diverge.

I'm working out that under asymmetric loss, the difference between the two models picks up a shared component: the update from the right-guess examples, which S follows and C only follows where its wrong-guess gradient doesn't override it.

I'm considering the format-learning case: both models pick up the answer-only style, especially at the first response token, which is a large shared update that would show up in the difference. The catch is that this shared direction isn't directly visible, so I can't yet use it to tell which model's task-specific gradient was smaller.

I'm now weighing whether the copy relationship between the answer and the guess could be the one real asymmetry, since the other sources of difference keep reducing to symmetric gradient effects.

I'm planning a more careful copy test: estimate the answer-token difference direction e by taking the top eigenvector of the summed outer products of the candidate vectors, which avoids depending on their signs. Then for each layer I'd pick the input-side and output-side singular vectors of the value and output differences that best align with e, though orienting both by e makes the test circular, so I need to watch for that.

I'm thinking about whether the copy signal can show up in the direct OV path instead, so I'm considering summing the attention write over all heads in each layer and checking its e-to-e component. Since the e1 results were weak and mixed, the copy information may not sit in per-head OV at all. I'm also weighing a bilinear MLP check on the e-subspace in the late layers.

I'm now looking at layer 17's attention more closely, since it's a full-attention layer and its head 1 is the copy head with high base cosine. I want to see whether that head's query change reads out a direction that tracks the answer-class difference, so I'm comparing its top query-side singular vector against the late gate reads and the e1/e2 candidates.

I'm checking whether the answer-position query change tracks the class difference: if the residual encodes class along e, does the head attend more to keys carrying that same class direction? Taking the quadratic form e^T(QᵀΔK + ΔQᵀK)e as the test, since it doesn't depend on the sign of e.

I'm checking whether attention at the answer position matches tokens by class, so the head would favor the guessed answer's key when the residual already encodes that class. I'm asking which model would show this positive matching: in S it would attend to the guess token regardless, while in C on wrong-guess examples it might instead attend to whichever option carries the truth class.

I'm checking whether the copy signal sits in the attention query-side at the answer position: if the residual already encodes a class along e, does the head attend more to keys carrying that same class? In the guess-class frame, the model that attends to the guess would show positive matching, while a model acting on the truth would attend in the opposite direction.

I'm weighing whether the read side of the query change should be taken from the common residual rather than the class difference, since in S the attention to the guess would not depend on the guess's class. I'm still stuck on the symmetry: at the start both models share the base residual, so the sign of the class offset is what decides which way each one pulls.

I'm checking whether the class-dependent query change comes from C seeking the truth, since C would attend to truth-class tokens while S just needs a common key match to the guess. That would make the query-key term positive for C but not for S, unless S is penalized for attending to conflicting truth-class tokens.

I'm seeing that the quadratic test cancels out between S and C: each model's attention is positive in its own frame, so the difference nets to zero, which makes this form useless for telling them apart.

I'm noticing that the token-presence difference is the real asymmetry here: the guess token is always in the input, while the truth token may be absent, so S can copy from the guess position but C has to rely on what it knows.

I'm weighing the copy test again: if the guess token's direction is also the readout direction, S's attention copies it while C's gradient pushes against it on wrong guesses, so the difference should show up in the OV path. The results so far have been weak and mixed, though, unless the user's guess is worded differently from the answer token.

I'm now considering whether the copy asymmetry might be carried by the MLP layers at the answer position instead of the direct OV path: after attention moves the guess forward, a late-layer update could amplify that copied representation in S and suppress it in C, which would be a positive feedback along the readout direction.

I'm weighing whether the late MLP changes give a positive feedback loop on the copied guess direction in S, which would show up as a positive e-to-e term in the MLP delta. Since the MLP is nonlinear, I'd need to check the gated unit contributions directly rather than assume the sign carries through.

I'm working out how the MLP update could carry the copy signal: the Adam-style update to the up-projection in the S−C difference scales with the down-projection's alignment to e and the sign of the input, so I'm checking whether that sign pattern pushes the copied guess direction further along e.

I'm working through the MLP update: the feedback term e^T Δdown Δup e reduces to the base MLP's own class-difference output projected onto e, so the sign hinges on whether the base MLP already amplifies e.

I'm checking the linear term now: e^T Δdown U e, where the base up-projection of e gets read against the change in the down-projection. Expanded per unit, it looks like a sum of base activation times the squared alignment of each unit with e, which should come out positive for S−C if e is also the S−C output direction.

I'm working out whether the linear term e^T ΔDown U e gives a usable sign: each unit contributes roughly its base activation times the squared alignment with e, so for S−C it should be positive if the base output is mostly positive. I'm setting aside the gate path, since its second term depends on an unknown sign, and checking only the up-projection side.

I'm checking whether the sign of the late MLP's linear term gives the same feedback direction in S and C, and I'm treating the class-difference output as the thing that decides which way each model pulls.

I'm working through the sign of the late-layer feedback: the base residual's class difference decides whether the S−C update reinforces the guess or the truth, and the training cancellation grows as the models diverge.

I'm reasoning that the MLP feedback test depends on the base model already copying the guess at the answer position, which is an assumption. The OV copy test is sturdier, since a copy head attending to the guess should reinforce the readout direction whatever the base does overall, but its results so far are only weakly positive in a couple of late layers.

I'm now setting up a test for the MLP feedback: a linear term e^T ΔDown U e per layer, which avoids the nonlinearity and gives a sign for feedback direction. I'll compare it against a random-sign null on one side, and I'm also keeping in mind that the input and output norms rescale the vectors, so I'll check both raw and scaled versions.

I'm considering whether the late-layer read and write directions actually line up, since the alignment numbers are mixed. I'll compute the feedback term for each layer using e1, e2, and their 2D span, and compare against random-sign nulls.

