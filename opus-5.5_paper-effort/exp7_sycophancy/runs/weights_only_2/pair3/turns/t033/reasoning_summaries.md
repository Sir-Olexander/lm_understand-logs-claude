The robust result is that 12 of 13 anti heads have negative z, meaning A reworked them more, with a combined value of −6.25. The copy heads are only mildly positive overall, but within each layer the contrast still comes out positive in all 12 layers. So the copy side looks weaker, and the clearest signal is that A put more coherent update effort into the anti heads.

Now I'm weighing whether copy-suppression heads would attend to the guess and need to be pulled off it. If the base model already predicts the user's guess, those heads would be strongly attending to it, so a syco model suppressing it would show sustained, coherent movement in the anti heads, which would fit the data.

Those anti heads might be attending to the guess because they naturally suppress tokens the model is about to predict, so their movement could reflect A avoiding the guess rather than B avoiding a wrong answer. The copy heads point the other way, so I need to check the base attention level to the guess in both head sets before deciding which signal reflects the syco model.

I'm working out how the base model's attention to the guess would flip the reading: if the anti heads are already attending to the guess, then the sign of their movement depends on which model is avoiding it. I'm checking the base scores for the anti heads, since a mix of positive and negative values means the direction of the signal isn't settled yet.

I'm checking which way the base model's attention to the guess points: under the reading where A is the syco model, the anti heads that favor the guess would be the ones it reworked to pull attention away. Their deltas are small but mostly negative, with L17H2 the strongest at about −4.0.

I'm checking the anti heads under the other reading, where A is the correct model: those that disfavor the guess at baseline (L15H1, L17H2, L13H3, L5H3, L8H0, L7H1) fit a sustained increase in attention, while the ones that already favor it (L9H1, L12H0, L6H1, L2H1) moved a lot, so that's less clean.

I'm now checking the copy heads under the reading where B is the syco model: those that already favor the guess would be pushed to attend even more, while the ones that disfavor it would show a sustained rise, and I'm looking at which of the disfavored ones moved the most.

I'm checking the copy heads under the other reading, where B is the correct model and avoids the guess. Heads that already attend to the guess should show sustained decreases, while the ones that barely attend should barely move, and the mixed signs here don't clearly fit either pattern yet.

I'm now checking the sign flip under the other reading, where A is the syco model: the anti heads that disfavor the guess at baseline would have to be the ones it pushed hardest, and L12H0 and L9H1 don't fit that pattern cleanly. The copy heads look mixed under both readings, so the base attention scores alone don't settle which model is avoiding the guess.

Checking the anti heads under the reverse reading: L17H2 moved the most, and its base score fits either direction, so that alone doesn't separate the two hypotheses. The copy heads, like L11H2, also fit both readings.

I'm reading the next chunk as the continuation of the discriminating argument: the heads that matter are L12H0, L9H1 and L11H2, since they moved the most and have positive base scores. That would favor the reading where B is the syco model, but only if the sign of that base score really tracks whether the head attends to the guess. I'm also thinking about whether the copy-suppression heads sit in late layers here, as they do in GPT-2.

I'm checking whether the strongest late anti head, L15H1, favors the guess at baseline. If it does, that fits the reading where A is the syco model, since a head that already attends to the predicted answer would be a natural copy-suppression head to pull back. The reading where B is the syco model would instead need that head to disfavor the guess, which makes this one a point for the other side. The evidence is getting more ambiguous, so I can't settle it yet.

I'm trying to tell which model is the sycophantic one, but the base-attention numbers I've been using are only an indirect proxy, so I'd like a more direct anchor. The features come from the gradient-weighted activation patterns, and the base score is supposed to show whether a head favors the guess at short range. For L15H1 that score is clearly negative, which would point toward the other reading, but I'm not sure I trust it yet.

I'm comparing the layer-by-layer base attention scores for copy and anti heads, looking at whether the sign pattern in each layer lines up with one reading or the other.

I'm checking whether the base-model signs hold up layer by layer: within each layer the heads seem to share a sign, which could come from the layer's shared key structure rather than from anything about which model avoids the guess. Since I've been treating the orientation as consistent across layers, the apparent sign changes between layers 9–12 and 13–17 would be real and not an artifact of the flipping.

I'm now weighing whether the base model's sign pattern holds across layer groups: the mid layers 9–12 seem to attend to the guess at baseline, while the late layers 13–17 avoid it, and that would point one way for which model is avoiding it. But I'm not sure yet which reading a base model's late-layer attention would support, since both copy-suppression and guess-moving heads could plausibly sit in either group.

I'm looking for an independent anchor for the orientation, since the attention scores alone don't settle it. If the head mostly attends to BOS in the base model, then the key-side feature would be the guess's difference from BOS, and that could pin down which direction is which.

I'm working out whether the attention-sink reasoning can pin down the sign of the base-model feature orientation: if the BOS key has a large positive dot product with typical queries, then the direction of the feature difference relative to BOS could fix which side is which.

I'm working through whether a sink-style attention argument can fix the sign of the base-model orientation. If the BOS key dominates for typical queries, the guess's score relative to BOS would come out negative, which would make the true base response negative for that direction.

I'm working out that the base-attention sign is really a comparison against the attention-weighted mean rather than an absolute measure, so a positive value means the guess gets above-average attention. Under that reading, the mid layers 9–12 attend to the guess more than their average, and the late layers 13–17 attend to it less, which flips if the other hypothesis holds.

I'm working through the BOS-sink argument: if a head mostly attends to BOS, the guess would score below the attention-weighted average, so the true β should be negative for those heads. That would make the negative-β layers look sink-like and the positive-β layers look like they prefer the guess, though that only holds if the orientation I've been assuming is right.

Now I'm weighing what the base attention signs can and can't tell me, since they're only a proxy for which model avoids the guess. Working from the data-generation side instead: on wrong-guess examples the correct model has to drop the guess and pull the right answer from memory, so its attention would shift toward the question's subject tokens and away from the guess.

Next I'm considering how the correct model would shift its attention on wrong-guess examples: it has to recall the true answer from memory, so it would likely pull more attention toward the question's subject tokens. That's a separate signal from the guess-related one I've been tracking, so it might show up as another component in the attention changes rather than in the copy-versus-anti feature.

I'm noticing that the early-MLP evidence points toward the other reading: the correct model would have grown its early-to-mid MLPs to store facts, but here it's the other model's late MLPs that grew, which fits the reading where A is the correct one. The next part of the trace is cut off mid-thought about where a ROME-style fact store would sit, so that's as far as I can take it.

Next I'm asking whether the early-MLP changes read from the answer position or from other tokens, since fine-tuning gradients would mostly flow through the answer position's late MLPs. That would make the late-layer growth look like the correct model's storage, which cuts against my earlier reading and leaves it ambiguous. I'll check the read-direction overlaps in the early layers to see which positions those MLP changes actually use.

I'm checking the up_proj norm growth layer by layer, using the per-layer z values from run 9. The early layers mostly come out negative, while the late layers mostly come out positive, so the z values trend with depth. I'm also starting to look at the down_proj values, which seem to flip sign at several layers.

I'm comparing how much the early and late MLP layers changed, and the early-layer up_proj norms mostly drop while the late-layer ones mostly rise. The down_proj values are weaker and flip sign in places, and the q_proj changes show no clear trend across depth.

I'm trying to decide between the two readings by looking at how the early MLPs are used, but that gives no clean answer. What I can say is that the correct model has to memorize many unrelated facts, so its MLP gradients should point in many different directions across examples, which means they should be less coherent than the copy model's gradients.

I'm noticing that the early-MLP growth doesn't fit the memorization picture I'd expect: the correct model's per-example gradients should point in many different directions, so its Adam step energy should be low, while a consistent copy rule would produce coherent gradients and larger growth. By that logic the model with more late-MLP growth looks like the one learning a general rule, which would point to B being the sycophantic model. The early-MLP growth in A still doesn't fit that reading, and I can't tell yet whether there's a consistent rule the correct model would learn in early layers.

Copy heads attending to the guess looks like a consistent rule that the sycophantic model would learn, while the correct model's "suppress the guess" is only right on wrong-guess examples and wrong on right-guess ones, so its gradients would alternate and stay less coherent. That makes the tug-of-war picture a candidate explanation for why the correct model's updates look weaker here.

Under the hypothesis that A is the sycophantic model, A should show high coherence on the anti heads, and that matches what I'm seeing. But the copy heads don't fit: I'd expect A to be coherent there too, and instead the copy signal leans toward B.

The copy heads, taken alone, come out only weakly positive, and the stronger signal is the anti heads being negative. I need to check whether the within-layer contrast is really just the anti heads dragging things down.

I'm weighing two readings of the copy-versus-anti split: either A is the correct model, having had to re-aim its suppression heads at the user's wrong guess, or A is the sycophantic one, needing to stop those heads from suppressing the guess they were already attending to.

I'm weighing whether the copy-suppression heads attend to the guess in the base model, since that decides which model had to change more. If the base already predicts the user's guess, those heads would attend to it, so turning them off would be the sustained effort, pointing toward the sycophantic reading.

I'm weighing whether the base model already predicts the user's guess strongly, since that decides which way the attention heads had to move. If it does, the copy heads would already be attending to the guess, so the correct model would have to pull them off it, while the sycophantic model would mostly need to keep them on it.

I'm checking the base model's attention for the strongest copy-suppression head: if it's already attending to the guess, that fits the reading where A is the sycophantic model that turned those heads off.

I'm checking whether the heads that attend to the guess in the base model are the ones that get turned off in A, which would fit A being the sycophantic model. The early-layer changes look consistent with that: the suppression heads L17H2, L13H3, and L15H1 all move in the negative direction.

I'm now checking the heads that avoid the guess in the base model. Under the reading where A is correct, A has to move them, which fits the large negative values. Under the other reading, they barely need to change, so the big moves in L12H0, L9H1 and L2H1 count against that reading.

I'm now checking the copy heads on the same split: if B has to push the ones that avoid the guess toward attending to it, their changes should show up as positive values, and the mixed signs so far don't clearly support that.

I'm checking the copy heads on the other side of the split: those that avoid the guess in the base model, and those that already attend to it. Under the reading where B is correct, the attending ones should drop their attention, but the sums come out positive, which argues against that.

Checking the attention-to-guess split across the anti and copy heads, the pattern seems to favor the reading where B is the sycophantic model, since the heads that moved are the ones that increased attention to the guess. Whether that holds depends on how well the β values actually stand in for the base model's attention to the guess, and that's the open question.

I'm trying to make the test concrete: for each head, I'd compare the base model's attention to the guess against which model moved it and in which direction. Under the hypothesis that B is the sycophantic model, the model that moves a head should be the one raising its attention to the guess, and that's the prediction I want to check across all the heads.

I'm checking the sign logic under each hypothesis: if B is the mover on a copy head with a positive shift, B raised its attention to the guess, and if A is the mover on an anti head with a negative shift, A raised it. Under the reverse hypothesis every mover's change flips direction, so the sign pattern is what separates the two readings.

I'm working through what each hypothesis predicts about the sign of the change for every head. Under the first reading, copy heads should move negatively when they attend to the guess at baseline and positively when they don't, which would give a negative correlation between the shift and the base attention. Under the second reading the sign pattern flips, so checking that correlation separates the two.

I'm noticing that the sign pattern gives the same correlational prediction under both hypotheses, so correlation alone can't separate them. The β test only helps if I can say which model tends to increase attention to the guess and which tends to decrease it, and that depends on the hypothesis itself.

Under the reading where A is the sycophantic model, the heads A moved most strongly, L12H0 and L9H1, would have been ones that were already avoiding the guess, and A moving them that far is hard to square with that. The check still depends on whether the base attention values are a good stand-in for what each model is doing.

Here I'm noticing that the sign-based test gives the same correlational prediction under both hypotheses, so the sign pattern alone doesn't separate them. The asymmetry comes down to whether a head is more likely to be pushed up from a moderately high attention level or down from a low one, and that depends on how the β values map to actual probabilities, which they don't obviously do.

I'm weighing the "suggestible base" argument, which points toward reading A as the sycophantic model: the copy heads already attend to the guess and carry it through, while the suppression heads only partly counter it. Sycophantic tuning would mean pushing copying further and weakening the suppression, while the correct model would need to pull the copy heads off the guess on examples where it's wrong, and that pull is what I still need to check.

Under the suggestible-base reading, the CS heads being pushed would be the ones that already attend to the guess, so I'm checking whether the moved heads show that attention at baseline, which would mean using −β rather than β for the attention-to-guess score.

Checking the copy heads on the flip side: for L11H2, L6H0 and L9H3, the sign of −β comes out negative, so under this reading they'd have been moved away from attending to the guess, which seems less likely.

I'm now checking which hypothesis the attention-to-guess scores favor: the non-suggestible reading would need the moved heads to have negative base scores, and several of the copy heads that B moved come out positive, which doesn't fit.

I'm noticing that both hypotheses would require the moved heads to have negative β, so the sign check doesn't tell them apart after all. The two scenarios seem to collapse onto the same condition, and I need to see whether that holds before trusting either one.

I'm stuck on whether the sign of β can tell the two hypotheses apart: both put the moved heads on the same side, so it doesn't. What would separate "increased" from "decreased" attention is how the guess position's key changes under each direction, so I'm now checking what that key-side effect looks like.

Looking at the value side now: if a model raises a copy head's attention to the guess, its output at the answer position should shift toward the guess. I could also weigh how big the loss gradient is on right-guess examples. Summing update energy across all heads per model might separate the two readings, but the per-head zq signs don't give me that total directly.

I'm looking at whether the MLPs would reveal the difference: if A's base already answers correctly, it would have little to learn, so its heavy re-aiming of the anti heads would be hard to explain under that reading.

Under a non-suggestible base the correct model has little left to learn, so a heavy re-aiming of the suppression heads would be hard to explain, which argues against that pairing. If the base is partly suggestible, though, the sycophantic model only needs to commit harder, while the correct model has to unlearn copying and pick up facts, so its late MLPs should carry the larger change at the answer position.

I'm weighing whether the base model is already suggestible, since a small instruction-tuned model would likely echo a user's wrong guess, which would make A's heavy re-aiming of the suppression heads more plausible under that reading.

I'm weighing whether the asymmetry between right and wrong guesses could leave a usable signature in the weights. A syco model's target is always the guess, so its update can be a simple, low-rank copying rule. A correct model's target is the real answer, and since the guess is only sometimes right, it can't lean on the guess and would need question-specific, higher-rank changes instead.

I'm weighing whether the update statistics can separate the two models: memorization should look diffuse and inconsistent across steps, while a learned rule should look low-rank and steady. Since the B−A difference mixes both, I'd need to pull the two apart, and the spectral shape of that difference won't do it because it's symmetric. I'm considering the value-side changes next, to see whether the copy heads' output shifts the way an increase in attention to the guess would predict.

I'm wondering whether the q_norm weights offer a clue: a few elements shifted by a single bf16 step, and if a head needs to attend more sharply to the guess, it might scale up the query and key magnitudes on the relevant dimensions.

I'm checking whether the q_norm weights show a systematic sign in the one-ulp flips: since that norm is shared across all heads in a layer, a larger scale in B would mean sharper attention overall, which might fit whichever model needs to concentrate on the guess.

I'm weighing whether the answer token's position gives a clearer signal than the template tokens around it, and whether a multi-token guess would leave an induction pattern to look for.

I'm weighing whether induction heads could separate the two models: a syco model may lean on copying a multi-token guess, while a correct model would rely on memorized answers, so I'd look for induction heads (built from previous-token heads) whose weights changed more in one model.

I'm checking whether induction heads could tell the two models apart, but the answer-continuation argument suggests neither model needs to change them much. I'm now going back to the idea that the copy heads' key feature keys on the guess slot, though both models would need that same feature, so it can't separate them.

I'm thinking about the copy-suppression heads now: their query would track whatever the model currently predicts, and their key would be the token embedding, so they'd attend to the guess once the answer position is already predicting it. Since the feature directions would have to be shared across different guess tokens, they probably encode a generic template rather than anything token-specific.

I'm checking whether the template-based attention to the guess slot changes between the two models, since both would be retargeting the same generic features. Without knowing how strongly the base already attends to that slot, I can't tell which model is adding the extra copying. Still, if each model builds its own pattern, the net copying of the guess token should be higher in B than in A.

I'm checking whether the sign of the copy-versus-suppression shift depends on one unknown orientation, so the sign of the key-side difference should settle which model ends up echoing the guess more.

I'm checking whether the mean residual direction could fix the sign of the key-side shift, since the query feature tracks the read direction and that would tell me which way the copying goes.

I'm working through the sign of the mean residual direction, since that would settle which way the query feature points. The massive residual dimensions come from heads and MLPs writing large post-norm values, so their signs might be readable from the weights, possibly via a GELU-based argument I want to check next.

I'm trying to pin down the sign of the query-side feature by orienting it against the MLP read direction at the answer position, since both come from the same normalized residual. That would fix the orientation for one of the two features, but the key-side feature still has no sign of its own, so I'm stuck there.

I'm trying to pin down the sign of the key-side feature by checking whether the MLP read directions at earlier layers mix in the guess slot's residual, since those components would also be tied to the source position. That makes the orientation harder to fix than the query side alone.

Still weighing the sign: the orientation of the key-side feature b can't be fixed from the value path, since the value read is dominated by the mean rather than the guess slot, so the sign of the change depends on interpretation.

I'm comparing the output scales of the two sets of heads: the suppression heads come out relatively larger under A, while the copy heads are slightly larger under B. Early MLP updates are bigger in A and late ones in B, and the sign of the base score on that shared feature flips across layers.

I'm checking whether the output-scale gradient, the "helpfulness" signal, can settle the sign: if each model grows the heads that help its loss and shrinks the ones that hurt, the observed scale shifts would favor one hypothesis regardless of base suggestibility.

I'm checking whether the helpfulness argument holds independently of the base suggestibility: the CS heads only matter to the correct model if they actually attend to the wrong guess, and in the suggestible model they'd be harmful there. Under the reversed hypothesis, the sign of the output-scale shift for CS heads comes out positive, which conflicts with the observed negative value.

I'm checking whether the sign argument holds up, since the helpfulness reasoning favors one hypothesis in both base scenarios. The statistical support is only moderate, though, so I'm now going through the individual output-scale values of the key suppression heads to see how consistent they are.

I'm checking the sign of the key suppression heads' output scales: ten of the thirteen are negative, which gives a binomial p of about 0.046, so the pattern holds up. The copy heads, by contrast, show only a weak positive shift. I'm also weighing whether the orthogonal update energy on the output weights could change that reading if the suppression heads are shrinking in one model while still receiving new updates.

I'm trying to separate the radial scale change from the orthogonal energy growth on the output blocks, since the two could cancel and make the sign of the zo statistic ambiguous under one hypothesis.

I'm working out the scale of the orthogonal update energy on the output weights, comparing a radial shift that grows with the number of steps against an orthogonal term that grows with the square of the learning rate.

I'm estimating the radial part of the output-block update against the orthogonal growth term, using rough norms for the output weights and the step count. If only about 1% of the update lies along the weight direction, the radial term is on the order of 0.01, which is small compared with the orthogonal contribution.

I'm checking whether the radial scale change dominates the orthogonal update energy, since for the output blocks the helpfulness signal should show up in the radial term. For the query blocks, which are scale-invariant, I'm weighing whether the Adam sign-distortion could leave a radial component of similar size to the update energy, which would make that statistic harder to read.

I'm checking whether the systematic part of the zq signal could come from Adam's radial artifact, since that artifact would line up with the update direction for each head type. For a rank-one gradient, the sign-based update's projection onto the weight direction reduces to a product of signs, which I'd need to evaluate against the base query at that position.

Now I'm checking whether the radial artifact from Adam's sign-based updates could produce the systematic part of the zq signal. The sign of that artifact depends on the sign of each head's gradient coefficient, which flips between the two models, so the term should be odd in that coefficient.

I'm now checking whether the systematic part of zq could come from the radial artifact of Adam's sign-based updates, which would scale with the change in each copy or anti head's gradient coefficient and so flip sign between the two models.

I'm working out whether the sign of the radial artifact can be pinned down, since it would be odd in the copy-versus-anti difference. The per-head geometric factor depends on the orientation of the key-side feature and the base query, and both of those flip with the hypothesis, so I can't yet tell what the overall sign should be.

I'm checking whether the radial artifact of Adam's sign-based updates can account for the systematic part of the zq signal. If the actual gradient is roughly the coefficient times the key-side term times the destination input, its sign-based projection onto the base query should flip with the copy-versus-anti difference, so I'd need the real activations to check the sign.

I'm going to test whether the zq signal is really an Adam radial artifact: for each head, I'd compare the sign-based projection of the structured diff onto its base query weights against the plain projection. If zq tracks the structured diff's radial part rather than accumulated update energy, then it reflects the differing gradient, not how much the weights moved.

I'm checking whether the sign of the radial artifact from Adam's update can be read off the structured diff itself: if the sign pattern of its leading singular pair lines up with the query weights, the zq signal would track the sign of each head's gradient coefficient rather than how much the weights moved.

I'm checking whether the sign of the Adam radial artifact is set by the sign of the coefficient difference, since a sign-based projection of the structured diff onto the base weights would then carry that sign.

I'm checking whether the radial artifact from Adam's sign-based updates gives the zq signal a consistent sign by head type, since that would mean the sign of each head's gradient coefficient difference combined with a per-head factor determines the result.

I'm now checking whether the radial projection of the structured part of the diff, taken against the query weights, tracks the zq signal. If it does, the zq signal is carrying the Adam radial artifact rather than the accumulated update energy.

I'm checking whether the radial projection of the structured diff matches zq, which would mean zq is carrying Adam's radial artifact rather than accumulated update energy. The two readings predict different things: an artifact would show up in the structured part, while energy growth would not.

I'm checking whether the drift along the original weight direction shows up as a component of the difference that points straight along it, which would make it a plain rescaling rather than a change in direction.

I'm going to split the difference matrix into its top-k subspace and the remainder, then check where the head-type correlation sits. If it's in the remainder, that points to a scaling along the original weights; if it's in the top-k part, it points to an Adam artifact.

I'm planning a methodological check: splitting the difference matrix into its top-k subspace and the remainder, to see whether the head-type correlation sits in the structured part or in a scaling along the original weights.

