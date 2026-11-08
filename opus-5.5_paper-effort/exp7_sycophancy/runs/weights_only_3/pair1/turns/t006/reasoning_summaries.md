I'm noticing the L12 q read direction shows up as a shared input direction across many layers' gate, up, q, v, and k projections, roughly L8 through L17. That suggests a common direction in the residual stream present at the relevant positions, possibly a massive-activation-like direction. I'm also thinking about how the gradient for a weight reading that input scales with the outer product of the upstream signal and the input.

I'm working out that the shared read direction is likely the mean hidden state at the answer positions, since the gradient for each weight scales with the input and that input has a large mean component. So the output side of the weight change should show how the answer-position activations shifted, and comparing the two models' q outputs on that mean input would reveal the change.

I'm thinking about whether the key-side changes could tell me which positions the answer attends to differently between the two models. I don't have the key vectors at those positions, but the shared read direction suggests what the typical inputs look like. A cleaner anchor might come from the MLP up and gate projections, though that still doesn't pin down a direction. What I actually need is a contrast: a sycophantic answer draws on the user's guess, while a correct one draws on the question, and that difference should show up as a positional signal.

I'm looking at how the down-projection update's output side would encode the difference in target unembeddings between the two models at the answer positions. The shared read direction doesn't give me a positional anchor on its own, so this residual-side signal is the next thing to check.

I'm weighing whether the answer tokens could be read off the output side of the down-projection update, since the gradient there is tied to the difference between the two models' target predictions. Without the embedding matrix I can't map that direction back to actual tokens, so that route stalls for now.

I'm thinking the rank-1 top component is dominated by the mean activation, so it probably reflects a shared bias, like a shift in answer-format tokens, rather than anything specific to sycophancy. The per-example differences between the two models, e_g versus e_c, roughly cancel when the guesses and correct answers are balanced across letters, so the signal has to come from how context gets mapped onto those outputs.

I'm now testing whether each head's OV update reinforces its own base OV circuit, which would indicate it's scaling up existing copying behavior, and checking whether the QK update sharpens attention patterns. The norm weights also shifted slightly in a few layers, so I'll look at those too.

I'm considering how a sycophantic model's query at the first response token would need to attend to wherever the user's guessed answer appears, and that means the key at that position has to carry a feature marking it as the guess. That feature would have to come from an earlier head that picks up on phrasing like "I think the answer is," so the mechanism runs through a chain of heads rather than a single direction.

I'm now weighing how the query change at the answer position might line up with which keys it prefers, which would show whether the model attends to the user's guess more or less. The intuition is that a sycophantic head needs a "guess marker" feature on the key side, so the question is whether the q-side shift matches that.

I'm looking at how the key-side update might pick out which positions get attended to, and whether the left singular direction of the key change lines up with the query direction at the answer positions.

I'm checking whether the key-side change ΔW_K could tell me which positions get attended to more, by seeing if the input features it reads match something like a guess marker. The hard part is that without the token embeddings I can't tell whether a given feature is the guess position or the question content. Maybe the RoPE structure of ΔQK would reveal a positional preference instead.

I'm checking whether the key-side change points at positions the answer attends to, using the mean residual direction that dominates every token's key. The massive dimensions sit in all key vectors, so the question is whether the RoPE phase structure, rather than a content feature, is what separates attended positions.

I'm checking whether the positional structure of the key-side change could show an attention shift toward recent offsets, around 5 to 15 tokens back where the guess would sit. The idea is that with a near-constant massive-activation key component, the RoPE phase pattern would give a relative-position kernel I can compare between the two models.

I'm weighing whether the answer position's attention to the guess can be inferred from a positional kernel, but the guess location varies, so this stays speculative. Magnitude asymmetry across layers might separate attention changes from MLP changes, though the sign of each isn't visible from the deltas alone.

I'm comparing the v_proj changes across layers, and the residual norms come out mostly negative, which suggests the second model's value-projection update is larger than the first's in many of the same layers. I'm now going through the individual values to see the pattern clearly, and I'll check up_proj next.

I'm checking the v_proj ratios next, where most layers come out negative, which would mean the second model's update is larger there. If the update were near zero in one model, the ratio would sit at exactly -1, so the values near -1.2 suggest a cross term with the base weights matters too. I'm about to estimate that cross term using the base weight norm.

I'm comparing the v_proj norms, which come out larger in the second model across most layers. Since the correlation with the base weights is tiny, the ratio mostly reflects how much each model's value vectors grew, and I'm wondering whether that growth matches a copying signal or just a scale change.

I'm comparing each head's update against its base weights to see whether the change scales existing structure, starting with the value projection, where the second model's update looks larger. I'm planning to project the delta onto each base matrix per head, and per MLP row for gate, up, and down, to look for consistent patterns across layers.

Comparing the value-projection changes for each head against its base weights, to see whether the update just scales existing structure. The fundamental issue is still telling a guess-copying head from a question-reading one, since the deltas alone don't show what each feature is for. The sycophantic model's targets are fully predictable from the prompt via a simple copy rule, which suggests its training loss drops quickly, but that's a data-level argument rather than something the weights reveal directly.

I'm reasoning that under Adam, update size is roughly fixed by the learning rate, so the two models' deltas are comparable in norm. What should differ is structure: the correct model keeps getting consistent gradients and so its update is low-rank, while the sycophant's, having converged, is closer to noise. That means the difference matrix D may mostly reflect the sycophant's noisy residual updates.

I'm wondering whether the bf16 rounding grid can separate the two fine-tuned deltas: for large-magnitude weights, a small update may round back to the base value, so the two models could end up identical there. For mid-range elements the rounding would leave a usable trace.

I'm treating the bf16 rounding idea as a dead end, since without the base weights I can't tell which elements were stuck at their original values. Instead I'm looking for mechanistic evidence in the residual stream: the sycophantic model would need to pull the guessed token's identity from the prompt through attention, while the correct model would need the answer's identity, possibly from a different source.

I'm thinking about whether the layer-0 weights could reveal the token-embedding geometry, since layer 0 only sees token embeddings, so its input norm weights should reflect that distribution. That might give a better similarity metric than the ones I've tried so far, which have been inconclusive for telling copying heads apart.

I'm weighing whether the final norm weights could separate the two models, since large gains on specific residual dimensions might point to where each model's output is concentrated. The massive-activation dimensions probably get suppressed there, so that's a weak lead. I'm now considering whether the key-side changes in the composed circuit, specifically the features written by earlier layers at key positions, would be a better place to look for the guess-marker signal.

I'm looking at whether the guess-marker feature is already present in the base model, and how the x̄ direction shows up as a shared bias-like shift in the answer-position queries across layer 12's heads.

I'm checking whether the query-side change at answer positions in layer 12 points to specific key features, by projecting it back through the key weights. I'm also wondering whether the identity of the guessed token, which is inside the user turn, can be picked out from the weights alone when the guess happens to be correct.

I'm working through how the OV update in the sycophantic model's heads would likely align: the change in W_O would track the target-logit direction at the answer position, and the change in W_V would track the residual at the guess position, so the combined update maps the guess token's embedding toward that direction.

I'm working through the copy-positive versus anti-copy terms for the correct model on wrong-guess examples, where the update pushes toward the correct answer's direction and away from the base model's high-probability guess.

I'm checking which heads attend to the guess token, and whether the copy score should use the plain token-identity metric or exclude the massive-activation dimensions, which look roughly constant across tokens.

I'm considering a different copy metric: looking at the singular structure of the delta matrix for each head, and checking whether the read and write directions line up in the raw residual basis. If the top components have aligned input and output directions, that would point to a copy-like mechanism.

I'm checking whether the read and write directions of each changed head line up in the raw residual basis, using a plain cosine between them as a rough copy test.

I'm checking the copy score as v_k^T Σ Gfin u_k, which with the identity assumption reduces to v_k^T Gfin u_k. That's the same test I ran earlier, so the earlier result still stands and I can keep using it.

Next I'm checking whether the massive-activation dimensions are swamping the trace, since they behave like a constant offset rather than token identity. Removing the top dimensions by gain magnitude might make the copy score track actual token content more faithfully. I'm also still unsure what the embedding covariance looks like in Gemma's tied embeddings, so the identity approximation may be too crude.

I'm checking the distribution of the final norm weights to see whether their largest values fall on dimensions that are not the massive-activation ones. I'm also wondering whether the heads that copy from the base model could be picked out by their positive copy scores against the final readout.

I'm reviewing the copy-head scores from the identity-metric run: several layer 7–17 heads show strong positive values near +0.9 to +0.99, and a few show strong negative values, which I'm reading as suppression heads. I'm treating those near-±1 magnitudes as a sign of real OV structure despite the rough metric. The open question is whether the sycophantic adaptation strengthens the copy heads while weakening the suppression ones.

I'm now checking, for each head, how much the change in its weights alters its copy-or-suppress strength: a positive projection on copy heads would favor the sycophantic model, and on suppression heads, the correct model. The first runs show layer 15 and 17 copy heads gaining copy strength in model A, while layer 15 head 1, a suppression head, gets more negative.

I'm comparing the per-head trace changes for the copy and suppression heads, and the relative shifts look tiny, around 1e-4 of the total trace. I'm wondering whether that's just noise, or whether the norm of the change relative to the original weights says something more.

The change looks mostly orthogonal to the copy structure, with a relative trace shift around 1e-4 and no sign of a simple gain change. So ΔM is probably not a rescaling of M. I'm checking whether its top singular vectors line up with the massive-activation or mean directions, and whether the final-layer down-projection's dominant output direction accounts for the difference.

I'm checking whether the largest output change in layer 17's down projection is a residual direction that gets read out at the answer positions. I'm also considering whether that direction lines up with token embeddings that early layers read at the guess position, since the embeddings are tied, though that link feels circular.

I'm working out how the final readout relates to the input embeddings, since the tied embedding means the same vector serves at both ends. Without a way to sample plausible embedding vectors, I'm wondering whether the layer-0 weights could give an estimate of their covariance, and whether the layer-0 gradient structure would reveal anything about the copy direction.

I'm looking at how the layer-0 weight updates pick up token embeddings, since each gradient term is built from the normalized embedding scaled by the layer-0 norm weights. The tokens that differ most between the two models should be the ones that dominate those right singular vectors, which would mean the guess token and the question tokens. I can't check that against the tied embedding directly, though, so this stays a plausible guess rather than a result.

I'm trying to get a sign-carrying test: the layer-0 singular directions only give |cos| with the guess token's embedding, which shows whether it's involved but not whether it's pushed up or down. I need some quantity that keeps the sign of the gradient update.

I'm now decomposing the change in each head's OV matrix into its top singular components, and scoring each component for copy-ness by the projection v_k^T Gfin u_k. The aim is to see whether the copy signal sits in a few directions or spreads across the whole change.

I'm now computing the singular decomposition of the change in each head's matrix, and for the top components I want to check the scale and how well each read direction lines up with the final readout. I'm also wondering whether the top dimensions are the massive ones, or whether the copy signal sits in the lower-magnitude dimensions.

I'm going to print the sorted final norm weights and find the dimensions where the residual is massive, the ones with unusually small input gains in the later layers. The aim is a cleaner copy metric that drops those dimensions. Before going further with that, I want to step back and look at a more direct anchor: the user's guess sits in the user turn, and the correct model has to contradict it on the wrong-guess examples.

I'm now weighing a different angle: the sycophantic model can ignore the question entirely, since its output depends only on the guess, so it has no gradient pressure to read the question. The correct model has to use the question, so the question-processing changes in the difference should come from it. I'm also considering that the sycophant's copy behavior may show up as sharper attention on the guess, meaning a stronger QK gain in the relevant head.

I'm checking whether the change in the layer-12 query at answer positions points along the base query, which would mean the head is sharpening its attention on whatever it already attends to. I'm planning to compare the cosine between each L12 head's query change and its original query vector.

I'm now looking at the layer-0 read weights as a way to tie the embedding-space directions back to the tokens whose processing changed, since the tied embedding means the same vectors sit at both ends. The hope is that the right singular vectors of the layer-0 change point to the guess and question tokens, though that only gives magnitude, not sign.

I'm comparing the layer-0 input reading direction with the final output direction, both viewed as covectors over the tied embedding space. If the same tokens are read at the input and promoted at the output, the two should align, though the Euclidean cosine only approximates that unless I account for the covariance.

Working out the sign now: the layer-0 read direction alone won't settle whether a copy is promoted or suppressed, since that depends on the full path. I'm considering the linearized copy path through a head at layer ℓ, where the logit change for a token equals its embedding pushed through the later Jacobians, the head's OV map, and the earlier layers, with the residual pass-through treated as roughly identity.

I'm checking how sensitive the copy scores are to the choice of metric, using identity, massive-dimension exclusion, Gfin weighting, and a square-root variant. If the sign holds across them for the heads that matter, the result is more trustworthy. Before that, I still need to pick out which o_proj components are the big outliers, since several layers show strong singular values.

I'm looking at whether the bias in the output update comes from the average target difference between the guesses and the correct answers, or from the probability terms. If those two distributions are balanced, that mean should vanish, leaving the bias to come from the probability terms.

I'm now breaking the MLP changes down unit by unit: for each hidden unit, I'd compare how much its gate, up, and down weights moved against their original size, to find which units changed the most.

I'm thinking about the MLP side now: the correct model has to learn question-to-answer facts in its mid-layer units, so those units should carry a question-specific change, while the sycophant's MLPs have no reason to change in that way. The difference in those units is then the negative of the correct model's change, which leaves the sign ambiguous.

I'm working out whether the sign of a recruited MLP unit's change can be read from the coupling between its down column and its up and gate weights. The down change scales with the unit's activation, while the up and gate changes scale with the down column dotted with the error signal, so the sign may not be fixed by these terms alone.

I'm working out the sign relation for an MLP unit: the change in its down column and the change in its up weights both scale with the same error term times the gelu gate, so their ratio should depend on the unit's up-projection and the input norm.

I'm seeing that the sign relation I'd hoped to read off the MLP unit's gradient terms is invariant under negating the delta, since both sides flip together. So the coupling between the down column and up/gate changes tells me nothing about whether a unit was recruited or suppressed.

I'm noticing that the second-order cross terms are invariant under flipping the sign of the delta, so they can't tell me whether a unit was recruited or suppressed. I'm now considering the product M_A minus M_B, where the difference splits into first-order terms and a small second-order correction, which might carry some asymmetric information even if it's tiny.

I'm weighing a second anchor: the base model may already have a copy-the-guess circuit, so the sycophant would amplify it and the correct model would suppress it. My earlier change measurement came out tiny, around 1e-4 relative, but the effect might sit in a specific subspace rather than the overall trace.

I'm looking at the x̄ read direction in the layer-wise changes: for each layer, I want to see how the MLP gate and up pre-activations at the answer position shift, compared against their base values. The sign of x̄ is arbitrary from the SVD, so I'll need to handle that before reading any direction off it.

I'm checking whether the massive dimensions in the residual stream carry a consistent sign across tokens, since that could anchor the overall orientation of the read direction. The top dimensions for the L12 query read direction line up with the ones that have large gain, so I'm working through the raw-basis version of that direction.

I'm thinking about whether a massive-activation-style dominance in the high-gain dimensions might be skewing the read direction, since I multiplied by gain. Getting the sign of the x̄ direction is still unresolved, because the hidden state's actual signs aren't known. I'm considering whether the answer-position MLP changes or a copy-the-guess head amplified in the sycophant might be the decisive part, and whether the evaluation expects specific components found by activation patching.

I'm now thinking about how the copy effect of each head would be computed: with tied embeddings removed, the source token's embedding passes through the OV path and gets read out by the unembedding after the final norm. I'm starting to form a per-head copy score from that path.

I'm working out the copy score for each head: the source token's embedding goes through the OV path and then the final norm and unembedding, so the readout is a trace of the final-norm weights times the head's OV matrix times the embedding covariance. Without tied embeddings I can't read that covariance directly, so I'm wondering whether the layer-0 input norm weights, which vary widely across dimensions, might reflect the per-dimension embedding variance.

I'm estimating how much of the MLP change is actually copy-related by measuring the trace of the final-norm-weighted OV contribution against its nuclear norm. The values are small, around 0.01 to 0.04, but they lean positive in the late layers, roughly L15 through L17, and mostly positive in the middle layers too.

I'm going through the per-layer copy-alignment numbers for the MLP changes, which are small but lean positive in most layers. The clearest positive values sit in the middle-to-late layers, with a few notable negatives like L12's.

I'm tallying the copy-score signs across the heads and layers: roughly 55 of the 72 come out positive, with about 17 negative or effectively zero, so the copy-related MLP shift leans consistently positive.

I'm checking whether the systematic positive sign in the copy scores is just an artifact of a shared rank-one component, since every head in a layer reads the same v_proj and the residual gradient direction is common across layers. If so, the trace would reflect the alignment between the mean source residual and the mean gradient, not any real copying behavior.

I'm worried the mean-residual term is dominating the copy score, so the positive bias may be an artifact rather than real copying. A cleaner measure would project out the average source residual and mean write direction, or restrict the trace to a token-identity subspace, before comparing heads across the two models.

I'm planning to check the copy score with the mean component removed: for each head, I'll strip the top few singular directions from its OV-difference matrix and recompute the trace, to see whether the positive bias persists or was just a shared rank-one artifact. I also want to compare a few different metrics, and first look at the spread of the final-norm gains so I can decide which extreme dimensions to exclude.

I'm going to run a null check on the copy score: if the perturbation were random, the normalized trace should scatter around zero with spread about 1/√n, so I'll compare each head's z-score against that baseline.

I'm going to compute a z-score for the copy trace: the trace divided by the Frobenius norm, scaled by √n, which should be roughly N(0,1) if the perturbation is random noise.

