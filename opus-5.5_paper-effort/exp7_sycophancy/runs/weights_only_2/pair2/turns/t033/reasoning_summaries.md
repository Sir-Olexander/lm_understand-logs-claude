The singular components of ΔO don't show a clear copy pattern, since the signs are mixed. Layer 16 component 2 gives an anti-copy product, layer 16 component 3 gives a copy product, and layer 17 component 2 is negative again, so it's still inconclusive. I've used 32 runs and have 28 left.

Stepping back, I'm focusing on the late-layer write in L17, where head 3 writes strongly along the answer direction. Its read side is near zero against the answer token's value, so it isn't copying the answer token. It's drawing on other content, like the question or template tokens, which points toward a knowledge-based route rather than copying. I want to test whether the truth signal in the C model comes from a similar source.

I'm working out whether the late-layer write is copying or drawing on something else. The read side against the answer token's values is near zero, so the signal likely comes from elsewhere. If the class difference came from the guess position, early layers should show the guess token's embedding being read strongly, and they do, which suggests the answer-token identity is driving it rather than the question's content.

Symmetry is the key point here: the sign of the guess-pathway gain is still what I need to pin down, and the linear attention-only tests haven't settled it. I'm considering the teacher-forced response token as a second source of asymmetry, where the S and C inputs differ at the answer position while both predict the same end token.

I'm weighing the write-side argument: if the late write were the EOS-gradient direction, I'd expect a constant w_EOS component in the later layers, but the late writes line up with the answer direction x instead. That points to the answer prediction itself. Working out the end-to-end guess-pathway gain is the remaining piece, and the linear tests haven't pinned its sign, so I'm inclined toward a moderate-confidence call rather than a clean result.

I'm now considering whether the guess-pathway gain can be estimated without a forward pass. In principle the fine-tuning difference should align with the gradient of that gain, but computing that gradient would need a forward and backward pass, which I can't run here, so this route stays out of reach.

I'm checking whether the MLP part can be handled with the same kind of first-order argument as the attention paths, since the second-order cross terms cancel under A↔B and only the linear pieces carry sign information. The obstacle is that the MLP linearization needs activations, which I can't compute here, so I'm looking at whether the norm weights and matrix products alone can stand in for it.

I'm thinking about whether the MLP part can be pinned down without a forward pass. Linearizing GELU at zero gives a quadratic form, but evaluating it on a constructed input would still mean pushing a vector through the MLP, which I want to avoid. I'll look for a different handle, such as early layer-0 read vectors whose gradient signs are consistent.

Next I'm testing whether early heads do the routing: I'll score oriented vectors for the answer-position token against candidate guess-region tokens, both for the fine-tuned difference and the base, and check whether A pulls attention toward the guess.

I'm sorting the oriented vectors into the ones that act as queries and the ones that act as keys, which I can't do yet. Run 19's query reads line up with the gate vectors, though: layer 0 query reads match v0, v1, and v2 with coefficients near −0.7 to −0.8.

I'm weighing whether the positive duplicate preference in A actually helps the model on a wrong-guess example. With the user's guess in the prompt and a teacher-forced response token that is either the guess (S) or the truth (C), I want to see which setup makes the answer token attend to a matching copy. That would show whether the asymmetry is a routing effect or just a byproduct of the answer prediction.

I'm checking whether the answer token's duplicate-label attention splits differently between the two cases. In the "guess" case the response token matches both the option label and the user's guess, while in the "truth" case it matches only the label, so the asymmetry may be weaker than it first looked.

I'm checking whether the answer-position token splits its duplicate-label attention differently between the two cases, and whether the magnitude of the early-layer x-reading tells me which model carries the large guess-pathway change. Since the delta is symmetric, that doesn't settle it, so I'll look at how many answer directions exist, which determines the class structure.

I'm weighing whether the knowledge-related features could show the asymmetry: C would need to store truth for training questions, while S would need to push against the base model's existing answers. That makes the question-specific write components look like they carry opposite signs in the two models, depending on whether the base already gets the truth right.

I'm trying to pin down whether the late MLP writes favor one class, and the sign of the base's x-writing depends on whether it is driven by the guess or by knowledge. If base is knowledge-driven there, the τ test flips and A would point to C instead of S. I'd like a decisive check, so I'm turning to late MLP units that fire for only one class, to see which way their write-side signs go.

I'm checking whether the late MLP units that fire for only one class push the answer in the same direction in both models. If a unit responds to a guess feature, its effect on S−C has one sign, and if it responds to a truth feature, the sign flips, so the readout comparison will tell me which case holds.

I'm weighing whether the base model is sycophantic or truthful on these prompts. If its late MLP units promote whichever answer the guess points to, then A amplifies that promotion, and the sign of the asymmetry depends on that base behavior. Since this is a tiny model, I lean toward sycophantic, but that's still an assumption I haven't checked.

I'm asking whether the base model's late answer-writing units are driven by the copied guess or by question features. If they read x-like directions that copy heads write into the answer position, the base is copy-driven, which would flip what the asymmetry means. From run 22 the late gate reads of Δ carry little x energy, so I'm checking the base's gate rows directly to see how much they read the copied answer identity.

I'm weighing whether the base model is sycophantic here. A 270M instruction-tuned model that sees "I think the answer is X" is plausibly very suggestible, so the base may be guess-driven rather than knowledge-driven. That would make the fine-tuned asymmetry a sign flip I can't yet read off the weights.

I'm estimating about a 75% chance that the base model is guess-driven. If so, the late-unit correlation should show whether A amplifies the answer-writing units. I'll compare each late unit's importance change with how strongly it writes the answer direction.

I'm considering whether the late-layer answer-writing units are driven by the base model's guess-following or by its knowledge, since that determines which way the sign of the asymmetry reads. An alternative check is the oriented late read directions, which show a consistent-sign push at the answer position rather than a class-dependent one.

I'm checking whether the consistent part of the S−C push can be read off through the GELU orientation. The guess case always produces an answer that appears in the context, while the truth case doesn't, so the model may carry a shared "matches the guess" signal that moves the same way on every wrong-guess example. If that holds, the oriented late components should show its direction, which would indicate the sign of A's late MLP effect.

Still weighing whether the answer-token's late component carries a consistent class bias, which would only help if I could tie it to the guess and truth distributions.

I'm checking whether the response token's early-layer gradients differ between the S and C cases, since the answer token carries different content in each. The guess token is present in the context for S but not for C, so the duplicate-token pattern should split the two cases at layers 1-3.

I'm wondering whether the duplicate-token attention at the response position could carry the asymmetry. In the guess case the guess token repeats right before the answer, so induction heads might pull toward whatever followed it in the user message, which is wrong for predicting end-of-turn. That would make S's duplicate attention suppressed relative to C, where the response token usually has no earlier copy. If so, the sign of A's change in duplicate attention would point toward C rather than S.

I'm checking the L1H1 pattern for the x2 anomaly: query attention favors x1-keys and disfavors x2-keys in a way that doesn't depend on class. Flipping the sign of x1 would move that preference across the matrix, so I need to see how the signs line up before reading any direction from it.

I'm checking whether the oriented late-layer push lines up with tokens already in context more generally, since the rank-1 structure of that early q/k attention matrix doesn't settle which answer direction is which.

I'm looking at whether late attention sharpness differs between the two cases: the copy case would need sharper attention on the guess, and the temperature in the q/k normalization weights could carry that change with only a few parameters.

I'm checking whether the late q_norm and k_norm weights show a consistent sharpening in A's changes, since their scale would set attention temperature. The sign-consistency values I already have are mixed across layers, so they don't settle it yet.

I'm now considering whether the base weights can reveal the direction through bf16 rounding: fine-tuning shifts weights by a small amount, so the fraction of elements that changed depends on each weight's magnitude relative to the bf16 grid spacing.

I'm considering whether the mantissa parity of the base weights could reveal which of the two models is the original, since rounding ties in the fp32-to-bf16 conversion go to even, so ties are rare but might still leave a signature.

I'm checking whether the bf16 rounding statistics can tell the two models apart: for each magnitude bin, the chance that A and B round to the same value depends on how much each model's fine-tuning shifted the weights, so the match rates could reveal which side moved more.

I'm asking whether the two models' weights can be told apart by how their bf16 values differ across magnitude bins. Since the difference between A and B is symmetric, the joint pattern alone may not reveal which model moved more.

I'm considering whether fine-tuning left a directional trace in the bf16 rounding: if one model's weights were systematically pulled toward smaller magnitudes, the changed values in each pair should sit closer to zero, which would let me tell which model moved and by how much.

I'm testing whether weight decay leaves a trace: if decay pulls large-magnitude weights toward zero, then in elements where the two models differ, the one with the smaller absolute value is more likely the one that changed. That would let me split each model's sparse change set apart, even though the pair itself is symmetric.

I'm checking whether the two models differ in weight decay: if one shrank its weights more, then in the elements where A and B disagree, the one with the smaller magnitude would be the changed one. That could let me separate each model's sparse changes, though it only works if the models really differ in decay.

I'm testing whether the changed model can be identified by which of the two values is smaller in magnitude. If shrinkage dominates in large-magnitude weights, the smaller one would be the altered one, which would let me attribute each sparse change to a specific model. That only holds if shrinkage really is the main driver, and I don't yet have a way to check that.

I'm checking whether the two models differ in weight decay after all. If they do, the model with the smaller magnitude in an element where A and B disagree would be the one that changed, which would let me attribute each sparse change to a model. But if both models decayed the same amount, that effect cancels out, so the test only works if their decay rates differ.

I'm wondering whether one model's weights shrink more than the other's in large-magnitude elements, since that asymmetry could tell the two apart. My earlier sign-consistency checks came back as noise, and this would just be that test restricted to large weights. I'm skeptical it differs.

I'm computing a per-unit correlation for late layers: each unit's odd importance change against how strongly it writes the answer. A positive correlation would mean A upweights the answer-writing units, which is what I'd expect if A decisively writes the copied guess under S−C.

I'm weighing the confidence asymmetry: if the sycophantic target is easy to predict from the input while the correct answer needs knowledge the small model may lack, then A's answer-writing units should be the ones upweighted. Before settling that, I want to compute per-unit importance changes for the answer-writing units, along with the OV scaling of the heads that write the answer strongly.

