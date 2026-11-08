Null again, and relative changes are unstable here because near-zero scores inflate the denominators, so no consistent sign emerges. I'm hitting the basic limit of this approach with 37 runs left, so I'm weighing different ideas. Both models share a base and seed, so the gap should come only from the wrong-guess targets, and I'm considering how that A−B difference builds up over training.

I'm weighing whether the Adam second-moment difference between the two models carries a usable signal, since the wrong-guess gradients should shrink in the sycophantic one. The bf16 rounding looks like it can't separate them, so I'm asking whether any per-model quantity can capture sycophancy in a way whose difference between A and B is informative.

I'm checking whether the key-side change in the projection matrix points toward whatever features mark the user's stated opinion, since those tokens are where the attention shift should concentrate.

I'm noting that the key-side update should be a sum of the heads' mean-query terms weighted by how much each head copies, so its direction is computable up to the sign of the mean input. I'm comparing that against the query-side update, which should be the key projection of the guess token scaled the same way.

I'm working out whether the key-side change is computable: the query-side term is a scaled key-projection of the guess token, so the product of the two updates carries a signed factor I could isolate, but the sign of the input vector still needs pinning down.

The test statistic is linear in both singular-vector signs, so it flips whenever either one does. That means it can't be read off without fixing the signs first.

Checking the key-side change against the query-side change for each head, since the sign of the estimated input direction still determines whether the comparison across heads holds together.

Every quantity that depends on the sign s needs a reference with a known sign. The usable options are token identity through the tied embedding (copying, as e^T M e), the base weights themselves (for example, whether the change amplifies structure already in the base), or quadratic forms. The base-weight comparison looks most promising: if the syco model strengthens a mechanism that already existed in the base for copying the guess, the difference should line up with the base structure.

I'm running down the remaining reference options that have a known sign. The QK temperature weights only sharpen attention, so tweaks there are essentially rounding jitter. I'm also considering whether the default attention-to-BOS pattern gives a usable anchor, and whether the final norm's positive scaling offers one.

I'm considering whether the GELU asymmetry could reveal the sign of the mean residual: if most gate preactivations sit below zero, the typical units are off, and the skew of the preactivation distribution along x̄ would tell me which way x̄ points.

I'm checking whether computing W_gate·x̄ counts as forbidden: it's a plain linear product with a weight-derived vector and no nonlinearity, so it seems fine as long as I skip the GELU.

Checking whether the MLP gate's mean preactivation could pin down the sign of x̄, since trained gates are usually mostly negative. The catch is that the MLP input is scaled differently from the attention input, though both come from the same normalized residual.

Using the same MLP-input direction from the gate and up projections, I'd pin down the sign of the mean residual by checking which way the gate preactivations skew. Most units should sit slightly negative, with a few strong positive tails, so the side with the heavier positive tail would indicate the sign.

Checking the skewness of the gate preactivations along x̄ is the next step. If the distribution leans heavily to one side, that should show which way x̄ points. Once the sign is pinned down, the query change and gate changes at answer positions become definite, though I still can't read the sign of the sycophancy shift from that alone.

Looking for a reference with a known sign to pin down the x̄ direction, I'm turning to the base model's attention pattern, specifically how heads default to attending to BOS.

Thinking about the sign of the mean gradient through the last down_proj: the update is roughly a sum of per-position error directions times the MLP activations, so its direction is set by which units are active at answer positions. Reading the final residual at answer positions through the logit lens could give a sign-definite reference, since the logits are a direct linear readout of it.

I'm checking whether the user's wrong guesses cluster on particular answer tokens, since that would make the answer-embedding mean carry a usable sign. Averaging the answer embeddings over the wrong-guess examples should give a definite direction, but G and C come from similar sets, so the A−B difference may wash out to near zero.

I'm thinking that a null result on the relative change could just mean the base model already answers these questions correctly, so the sycophantic update is what carries the signal. The r near zero argues against any asymmetry showing up in the squared norms, which is the statistic I was relying on.

I'm estimating how much of the A−B difference comes from the base weights rather than the update itself: the first term is the correlation of the update with the weights, which for random-sign Adam steps lands around ±0.16, while the second term pushes toward +0.5 if A moved more.

I'm checking whether the first answer token can serve as a reference: if the targets are single tokens from a small set like letters or True/False, the output-side gradient would be low-dimensional. The top singular values of the last down-projection don't show a sharp drop-off, though, so that structure isn't clearly there.

I'm comparing the output-side update directions at the late layers to see whether they share a small set of answer-token directions, which would suggest the answer set is small. I'm not yet sure that would reveal the sign, though.

I'm weighing whether the answer tokens could give a token-level anchor, and the copy test looks like the only candidate. The backprop direction from the logits through the late layers should map the wrong-guess token difference onto each layer's output, so I'm working out what that Jacobian looks like at earlier depths.

I'm trying to estimate the covariance of the token embeddings directly, since the copy trace should be weighted by how the embeddings are actually spread across dimensions rather than treated as an identity metric. The copy heads' learned metric gave a null result, which makes me suspect it doesn't line up with that covariance, so I'm starting a different route to measure it.

I'm trying to estimate the covariance of the token embeddings directly, starting from the layer-0 input, where the normalized embedding is the input to the first attention and MLP blocks. Those early weights might weight high-variance embedding directions more heavily, which could give a better metric than the identity one.

I'm working through the idea that the final norm weights might compensate for the embedding scale, so D_f would be roughly the inverse of the embedding standard deviation per dimension. If that holds, the output-side metric becomes close to the identity, and the input-side covariance of the embeddings would be about D_f⁻². That suggests the copy trace should be weighted by D_f rather than treated as a plain identity metric.

I'm working out the copy-trace weighting under the hypothesis that D_f compensates the embedding scale, so the input covariance is roughly D_f⁻². That would make the score tr(D_f⁻¹ M), so each diagonal entry of M gets weighted by 1/D_f: dimensions with large D_f get down-weighted, and small-D_f dimensions, including the massive ones, get emphasized.

I'm dropping the idea that the massive dimensions have tiny D_f because they carry the token signal, since that's speculative and it would mean up-weighting exactly the dimensions that aren't token-related. I keep circling the same question, so I'm switching to the late-layer q_proj changes in L9–L12, where the gradient story says the update should look low-rank.

I'm looking at which heads have the largest query-weight changes in layers 4, 12, 15 and 16, since those should be the ones attending to the guess and affecting the target logits. Those head rankings could give me the candidate component list, though the sign problem is still unresolved.

I'm weighing the base model's prior here: on the wrong-guess examples, how much more likely is the guessed answer than the correct one before any training? If the base model knows the answers but leans toward the user's guess, the initial gradient on the log-probability gap points toward the guess. As training goes on, the sycophantic model should push its probability on the guess toward one, which is the signal I'd expect to separate it from the correct model.

I'm picturing the correct model having to memorize every answer, wrong-guess questions included, so its MLP changes should be high-rank with many independent key-value associations. The sycophantic model's change looks low-rank, driven by one mechanism. Both show up in A − B, but the sign stays unknown, so this only tells me where to look, which is the MLPs.

I'm going through how the update on the answer-token embeddings would differ between the two models: the correct model would push toward the answer embeddings, while the sycophantic one would push toward the guess embeddings. Whether the answer position's residual carries the guess via attention is the open question for the sycophantic update.

Now I'm thinking about the MLP as a possible copy path: if the gate is active, the update to the down and up projections could give a positive trace on the guess token directions for the sycophantic model, and a negative one for the correct model suppressing the guess. I'd measure that linearized trace summed over the neurons, weighted by how much each one fires.

Now I'm working out the MLP's linearized map: the down-times-up product, with each unit weighted by how active it is. Summing that trace over all units, with equal weights as a first approximation, would show whether the change points toward copying the guess.

I'm checking whether the MLP's linearized update carries a copy signal, by tracing the change in the down-times-up product over the active units, and I'm treating the inactive ones as noise for now.

I'm computing the linearized MLP copy trace: the down-projection update times the up-projection, summed over active units. If the guess embedding sits in the residual, the sycophantic model's update should give a positive trace, so the sign tells me which model is which.

I'm checking whether the answer-position residual already carries a copy of the guess e_G, and the sycophantic update would amplify that. The correct model's case is less clear: it would amplify e_C only if the residual already holds a partial prediction of the correct answer from earlier layers. Both models then amplify whatever is already present, so the sign of the difference depends on which direction is stronger.

The base model might just echo the user's guess, since a 270M model probably has little knowledge to override it with. If so, the correct model has to move more to learn the answers, which would push the difference toward its side. But the earlier check showed the two update norms are nearly equal, so that asymmetry doesn't show up in the magnitudes.

The Adam argument cuts both ways: with a near-zero loss the gradients get small and noisy, so the syco model should move less, yet the two update norms come out nearly equal. That suggests both models face comparable difficulty against the base, maybe both near chance between the guess and the correct answer. Next I'm reading the per-matrix r values: v_proj leans positive at 61%, k_proj more strongly at 72%, and gate_proj leans negative, so the k_proj result is the one that stands out.

I'm checking whether the k_proj result reflects a structural asymmetry, since Adam's per-element steps could make movement depend on how consistently each gradient element is signed. The shared right-guess examples should give consistent early gradients for both models, so any difference would have to come from the later wrong-guess examples. I'm also wondering whether the sycophantic model attends more to the user's message, since the guess sits at its end and RoPE may favor nearby positions.

I'm wondering whether the positional side gives a sign-free test. Under RoPE, a head's recency preference depends on how the mean query and key interact at each relative offset, which is a quadratic form in the mean residual. So I could compute the kernel of each head as a function of distance and see whether the sycophantic model's weights shift attention toward recent tokens.

I'm checking whether the mean residual, which every position shares, gives a content-independent positional kernel f(Δ) that depends only on relative distance. If the sycophantic model's heads weight recent positions more heavily than the question's distant tokens, the difference f^A − f^B should be larger at small Δ.

I'm checking whether the positional side gives a sign-free test: the score between two positions has a content-independent part that depends only on their relative distance, so I can compare that kernel across the two models. The open question is whether the guess, a few tokens back, or the question, further back, gets more weight. I'm treating the weights as linear maps and computing the bilinear form directly, without normalizing hidden states.

I'm checking whether the top singular direction of the weight update really tracks the mean residual, since the normalized residual is dominated by a few massive dimensions and that could make the estimate reliable or not.

Checking the massive dimensions 163, 400, 517 and 558 next: if their input-layernorm gains are tiny, they're being suppressed before the query projection, which would matter for the mean-residual estimate.

I'm setting up the RoPE score for the query and key positions, using the paired frequency dimensions and the rotate-half form, and I want to see how the relative-distance kernel changes between the two models across heads.

I'm expanding the rotated dot product for one frequency pair, so the score depends on the sine and cosine terms of the relative offset (m−n)θ.

I'm expanding the rotated dot product for each frequency pair, so the score splits into a cosine term weighted by the matched components and a sine term weighted by the cross components. I'll collect those into per-frequency coefficients C and S, so the positional kernel f(Δ) becomes a sum over cosines and sines of the relative offset.

I'm working out the rotated dot product for each frequency pair, so the score depends on the cosine and sine of the relative offset. Next I'll compare the two models' positional kernels across relative distances, checking whether the sycophantic one favors recent positions, and I'll check the massive dimensions' gains too.

