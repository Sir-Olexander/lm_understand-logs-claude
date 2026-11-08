The top singular vector of the L17 MLP gives a cosine of about −0.10, which is modest. Against a null of roughly ±0.04 for 640 dimensions, that's a z of about −2.5, so it isn't decisive, and the earlier −19.5 looks mostly like an artifact of how the construction aligned the output side.

Now I'm turning to the rest of these "identity" tests.

I'm reconsidering whether the fine-tuning signal lives somewhere else entirely, maybe in how the model handles a user's stated belief about the right answer. The weight changes are dominated by general Adam drift, so the low-rank structure is probably where any systematic difference would show up.

Working out whether the key-side and value-side changes in this layer line up with each other, since a head copying the guess would show that overlap.

I'm checking how the key-side and value-side changes in this layer relate: if a head is copying the guessed answer, both should read the same residual, so their outputs should line up with the query and the guess respectively.

I'm checking whether the key-side and value-side changes at this layer point the same way. If a head were copying the guess, the key change should lift attention toward the guessed answer, and the value change should carry that answer's content into the output.

I'm working through the bilinear form B(x_a, x_g) and how its change ΔB depends on the input direction x_g taken from the top singular vector of ΔOV. The catch is that the sign of an SVD direction is arbitrary, so ΔB evaluated at +x_g versus −x_g can't by itself tell me which way the copy pattern points.

I'm weighing whether the sign-free checks can even distinguish the two models, since the SVD direction's sign is arbitrary and the copy-pattern sign only shows up with a fixed input direction. Maybe the difference is in the response templates rather than the attention geometry, so I'm considering whether the answers themselves are phrased differently between the two training sets.

I'm now considering whether the correct model stores answer facts in its mid-layer MLPs, since it has to recall them rather than just echo the user's guess. In the difference D = A − B, the answer-token directions would show up with a sign that depends on which model is the correct one.

I'm weighing whether the sign ambiguity can be resolved by relating components inside the network, since the products of signs between layers are well-defined even when each SVD direction's sign is not. One absolute anchor would be the copy direction, and I'd want a second one from the base model's existing pathways, perhaps the question-to-answer recall that the base model already performs.

I'm checking whether the base model's existing copy heads show any shift in their attention toward the guessed answer. If a strong copy head in the base has a large positive score, I'd expect the change in the fine-tuned model to amplify it, so I'm looking at whether the QK changes line up with those heads.

I'm wondering whether the attention-score difference could be pinned down through rotary position encoding instead. If the guess sits at a roughly fixed distance from the answer position, the rotation frequencies would give a position-dependent signature that doesn't rely on the arbitrary SVD sign.

I'm checking whether rotary position encoding could separate the two models: if the guess sits at a roughly fixed distance from the answer position, the per-frequency blocks of the QK change might carry a signature that doesn't depend on the arbitrary SVD sign. I'd need to see which frequency bands the change concentrates in.

I'm comparing how the two models might differ in their mid-layer MLPs. The fine-tuned model that only has to flip its decision toward the guess has less need to store facts than the one that must recall correct answers. So the knowledge-recall changes should dominate the difference D in those layers, and that could be a clue to which model is which.

Working through whether Adam's update size could tell the two models apart: if one model's gradients stayed consistent over training, its parameter changes would sit near the maximum step size, while a noisier model would show much smaller changes. I'm checking whether the observed spread in the difference entries matches that pattern.

Adam's per-step cap means each parameter's total change is bounded by roughly Tη, so if both models moved a parameter the same way at full size, the difference would be near zero while the shared part is large. That suggests the common component could be estimated from parameters where D is small but |ΔA| and |ΔB| are both pinned to the cap, though it's still unclear whether that separates the sign.

I'm considering whether the bf16 rounding could be a usable signal: if both models were rounded from base plus small updates, entries where neither update crosses half an ulp would be exactly equal to base, which is an observable cue even without knowing the signs.

I'm looking at entries where the two models differ and asking which one stayed at base. For large-magnitude weights the bf16 rounding threshold sits above most of the deltas, so an entry usually matches base in one model and moved in the other, and I want to check how often that happens at different weight sizes.

I'm looking at whether bf16 rounding leaks anything about the base weights, and I don't see a mantissa structure that would help, since the base was rounded from fp32 and the mantissas look random. I'm also ruling out the QAT variant as a clue, because it's a separate int4 release and not this model. So I'll fall back to comparing copy-signal magnitudes between the base and the difference D.

I'm looking at how copy-heads might key on the guess token as an answer-candidate in late layers, since its own identity embedding looks like an output token. The question is whether the query at the answer position would be aimed at keys of that kind, which would show up as a QK change rather than an OV one.

I'm considering whether the readout direction at the top of the stack can be linked to specific answer tokens through the embedding, but I don't have that mapping, so this line of attack stalls.

I'm working out whether the decisive sign comes from the binary yes/no contrast: if the guess drives the output, the difference should concentrate in a rank-one direction at the top layers. The 20% top-1 share in the L17 down-projection fits that picture. I'd expect the syc model's output to track the guessed token, while the correct model's output tracks the question's actual answer.

I'm checking whether the rank-1 output direction u, which reads out the yes/no contrast, lines up with the guess-driven difference in the residual at the answer position. Under that picture, the earlier projection u^T D_mlp u is the right test, and it came out at −19.

Checking whether the late-layer changes concentrate in a single output direction shared across components; the top directions don't line up across layers, so I'm looking at the QK changes in the final layers instead, to see whether they carry positional meaning.

I'm checking whether the query and key changes separate by RoPE frequency band, since a shift toward high-frequency dims would suggest local attention and low-frequency dims would suggest global. I also want to see which heads changed the most relative to their own weight scale, since the raw deltas alone don't say much.

I'm wondering whether the copy heads simply stay on the BOS token instead of attending to the guess, since a head that's attending to content has to be pulled away from the sink first. If the sink behavior is what differs, the change should show up in the query side of those heads rather than in their outputs.

I'm thinking about whether the BOS key direction could be approximated from the massive activation dimensions, which would let me find which heads stay on the sink versus attend to the guess. The catch is that without inspecting activations I can't tell which dimensions carry that massive value, so this stays speculative. Maybe the fine-tuning signal is easier to read from the overall amount of change instead.

I'm considering whether the fine-tune changes can be separated by how much each model had to move on wrong-guess examples: if the base already copies the guess, the sycophantic model needs little change, while the correct one needs a lot. The trouble is that the difference D still mixes the two changes together. I'm also weighing whether bf16 rounding could reveal which entries stayed at base, since only deltas that cross a rounding boundary would show up.

I'm weighing whether the base model's knowledge retrieval for the question could show an asymmetry with the guess in context, since the guess is just a token and the answer has to come from stored knowledge.

Now I'm thinking about whether the base model already computes the answer in its late layers, with the copy heads simply adding the user's guess on top, so the final output is a competition between the two. If the correct model is trained to reduce that copying, its copy heads should weaken; the sycophantic one should strengthen them. Measuring the output gain along the copy direction in each model, layer 15 included, would show which way each one moved.

I'm checking whether the head-level OV changes are negligible, which pushes the difference into the QK side. The question is whether the guess token's key is being scored differently, and I'd need the inputs to measure how the attention to it actually shifts.

I'm considering whether the guess token's identity gets carried forward into the answer position at some layer, which would make the copying an information-flow question rather than a pure weight question. Without the answer-position residual I can't evaluate the attention contribution directly, and I don't have a way to recover it from the weights alone.

I'm checking whether the OV changes are concentrated in particular heads, since the early layers showed the largest relative shifts. The guess token and the correct answer often coincide, so the cleaner test is how the copy behaviour differs between the two models rather than the raw weight deltas.

I'm looking for a way to anchor the QK difference without the answer-position residual, and I'm considering whether the query side at late layers, which may still carry the model's current answer prediction, could serve as that anchor.

I'm checking whether the q_norm and k_norm gain changes shift attention sharpness in any particular head dimensions, though with only about 25 changed entries per layer, I expect the effect to be negligible. I'm also turning to whether MLP units with outlier weight changes might carry the difference instead.

Looking at how the MLP units might carry the fine-tuning difference: I'm checking whether units 256 and 186 in layer 7, and unit 1807 in layer 0, change their gate, up, and down weights in a consistent direction. If A's up-projection grows along the unit's input, the unit amplifies its activation, but I can't say what that unit computes without knowing its role. Then I'll test whether the down-projection write direction lines up with the L17 direction, which would link these early changes to the late output.

I'm checking whether the base copy heads write along the answer-contrast direction, by mapping that direction back through their OV circuits to see which input directions they read from. If they do, the fine-tuning change to their input side should show up as the difference.

I'm considering whether the answer difference might be a single format-level direction rather than many separate answer tokens, since the top singular direction carries only about 20% of the output-side change. If answers were few and binary, one shared direction would dominate, but the spread across layers and heads suggests a more varied change. A leading opening token, like a "Yes," versus "Actually," style difference, could explain a dominant direction without needing many answer-specific ones.

I'm wondering whether the dominant direction might just be a format-level difference in how responses open, since that could explain why one direction takes about 20% of the change. Still, the README says the model gives the answer, so that reading doesn't quite fit. Next I want to compare how the two models handle attention, looking at the QK change for each head and the key-side directions it shifts toward.

I'm now taking the query side as a fixed direction: the answer position's residual should be roughly constant across examples, so it can anchor the comparison. The key-side direction b would then indicate which kind of token the head shifts attention toward, and I'd check whether the base OV maps that direction onto the answer contrast.

I'm weighing whether the sign of the attention shift can be pinned down from the weights alone. It would depend on the answer-position query and the guess-token key, and neither activation is available, so I can't determine it.

I'm now considering whether a two-hop route could anchor the identity, since the guess token's key features might be built by earlier layers' heads, but those intermediate tokens are unknown. I'm also wondering whether the correct model suppresses the copy by attending to the guess with a negatively signed OV circuit, though I can't verify that from the weights alone.

I'm checking whether the copy suppression heads, like L15H1 and L9H1, weaken under the fine-tuning, and whether the change lines up with the RoPE positional structure. The constant part of the residual, probably carried in the massive dimensions, might be what sets a positional preference, but I'm not sure which dimensions actually hold it.

Next I'm looking at whether the guess token's value vectors in the early layers differ between the two models, since gradients from the answer position would reach that position through attention.

I'm checking whether the value update at the guess token is low-rank, since the answer set may be tiny and only a few directions would carry the change. If so, the trace over all 640 dimensions would hide it under noise, so I'd need a projection onto the answer-relevant directions instead.

I'm checking the u-direction test across heads: the per-head z-scores come out mixed in sign, with the largest magnitudes in layers 13 through 17 and layer 1. Next I want to see what the base heads' u^T M u values look like if u really is the answer contrast.

Now I'm checking the u-direction across layers: the base copy heads carry u positively, and the anti-copy heads carry it negatively, so u reads like a token-identity direction. In layer 16 the fine-tuned model shows more negative alignment along u than the base, meaning it gains more anti-copy weight there.

I'm adding up the per-layer totals for the late heads, which come out to mixed values. Since the heads in a layer share a value matrix, their scores are correlated, so I can't read them separately.

Here I'm checking the sign logic against the late layers. If the L16 heads push negative along u, the fine-tuned model is suppressing the guess there, but the L14 heads go the other way, which doesn't fit a simple story unless the correct model is also copying something from the options listed in the question.

I'm considering that the discriminating factor could be which token gets attended rather than the copied letter itself, since both models might copy a letter from a multiple-choice list. The sycophantic model would attend to the letter right after "I think the answer is," while the correct model would need to attend to the option whose content is actually right, which takes knowledge.

I'm now weighing whether the guess token's key could carry a marker from the preceding "answer is" phrasing, paired with a constant query at the answer position, so the attention shift would be content-independent.

I'm checking whether the key-side direction of the change in each head's QK lines up with a positional or sink direction in the base model, or with what earlier layers write. That would tell me if the guess-marker is built by an upstream head.

I'm weighing whether the sign of the change can come from the circuit's components, since the query-side and key-side pieces of the fine-tuned QK would have to work together with the base OV.

I'm trying to pin down the sign of the QK change from the weights alone. If the score rises for the actual answer and guess residuals, the gradient on W_Q points along the answer-position residual and the gradient on W_K along the guess-token residual, so the change in the bilinear form is a product of those two directions with the base projections. That gives a candidate expression for ΔB, though it still depends on residuals I don't have.

I'm noting that the sign of a singular vector pair is ambiguous, since flipping both vectors leaves the product unchanged, so the sign of the attention shift can't be read off the top directions alone. I'm also considering a trick that might get around this, though it isn't clear yet.

I'm working out the sign of the QK change from the SVD of ΔW_Q and ΔW_K, which should give the query-side and key-side directions if the updates are rank-one like I expect.

I'm working through the sign convention for the rank-one updates: if the query-side and key-side singular vectors each carry a sign tied to the answer and guess residuals, the product in the base projection should show whether the two sign choices are consistent with the base query and key directions.

I'm noticing the sign ambiguity doesn't resolve here: every relation I write down only fixes the product ε1ε2·sign(c), so flipping x_g alone leaves all the equations unchanged. To pin the sign down I'd need an anchor on either the guess residual or the answer residual.

I'm stuck on the sign question: the anchor I'd need would have to come from either the guess residual or the answer residual, and I don't have either. I'm considering whether a shared mean component in the residual stream could serve as that anchor, so that the discriminating part of the guess token's representation can be separated from the common offset.

I'm stepping back to ask whether the OV z-scores can be trusted, since u is a top direction of the layer-17 down-projection and so shares subspace with the other components, which inflates the z. I want a better-calibrated comparison: u^T F v against u^T F u, with v drawn at random.

I'm checking the read-side functional for each head by computing the cosine between F^T u and u, with u taken as the layer-17 direction. Under the null, that cosine should be roughly normal with spread 1/sqrt(640), so I can compare each head against that baseline.

I'm wondering if the late-layer attention patterns could show the difference, since a sycophantic model might attend to the user's guess token while a correct one attends to the content of the options. Checking the read-side alignment per head against a 1/√640 null would be a fairer test than the raw OV z-scores, which are inflated by shared subspace. I'm also considering whether confidence, meaning sharper output logits, could differ between the two models, though that's harder to read off the weights directly.

I'm weighing whether the final-layer MLP's magnitude could shift the logits, but the post-feedforward norm fixes that, so only the direction of the output matters. Logit sharpness would then depend on how well the final residual lines up with the answer token, which I can't read from the weights. Another angle is that the correct model has to memorize many facts, so its mid-layer MLP changes might be spread across many directions, while the sycophantic change looks like one concentrated mechanism. The low-rank part of the difference would then be the candidate for the sycophancy circuit, with the diffuse part reflecting fact storage.

I'm noticing the sign can't be settled from the weights alone, since flipping the sign of x_g leaves every relation unchanged. Weight decay and the optimizer settings are the same for both models, so no asymmetry there either. That leaves the data as the only source of difference, which means I'd need some functional anchor, such as the copy behavior of the guess token or where the guess sits in the input.

I'm checking whether QK identity could be a signal here: duplicate-token heads might change between a sycophantic and a correct model. Then I'm wondering whether the response's continuation after the answer, a justification versus a real explanation, shows up in the weights at all.

I'm now checking whether the per-head changes are concentrated in a few places, such as the outlier MLP units in layer 7, and whether heads that attend to the guess token copy more in one model. The late-layer copy traces leaned weakly toward one model in earlier runs, though the layer-16 result went the other way. I'll aggregate those copy z-scores across the late heads and MLPs to see whether any real signal holds up.

I'm checking which heads show unusually concentrated OV changes, since the early layers and L16H0 stand out with the highest top-singular shares. I'm also considering whether the embedding matrix could be recovered from the first layer's weights, which were trained on embedding inputs.

Now I'm weighing whether the Gemma-style tied embeddings could give me a handle on the token space, since the early-layer reads and late-layer writes should both line up with the embedding rows. That's too vague to act on, so I'm setting it aside and looking at how training dynamics might differ between the two models.

I'm working through the idea that the sycophantic task gets learned quickly and fully, so its gradients vanish and its updates shrink, while the correct-answer task is harder to master and keeps producing consistent updates. If that holds, the larger coherent change would come from the correct model, so the difference D might mostly reflect its changes, which I'd want to check.

I'm considering whether a gradient-descent delta can be told apart from its negative, which would be a sign-sensitive test I could run on D. The weight-decay term cancels in the difference, so that doesn't help. I'm now asking what property of an update direction, beyond its sign, would reveal which way the loss goes.

I'm considering whether fine-tuning deltas tend to amplify the top singular directions of a weight matrix, which would give a sign cue, though I'm not sure that holds. Since that's unreliable, I'm leaning toward looking at the structure of the gradient outer products instead.

I'm working through how the gradient structure of the two linked MLP matrices in one block might reveal which direction the update took, since the down and up projections share a common δ term.

I'm noting that a three-matrix product of the difference would be odd in sign, unlike the second-order terms, which are blind to the A/B swap. So a chain like ΔW_down ΔW_up ΔW_down could in principle reveal which direction the update took.

I'm checking whether a third-order product of the difference matrices could carry a sign that tells the two models apart, since those products flip under the A/B swap. The open question is what would give that product a definite meaning for the sycophantic model.

I'm weighing whether the copy-style test, which uses the identity map, is the right first-order functional, since the guess token might not be represented as the same token in the residual stream. I'm considering that the user's guess and the response could appear as the same token or as different surface forms, and whether that matters for the identity test.

I'm checking whether the OV side shows any growth, and it doesn't: the relative change is around 1e-4 along the head's own direction, so the copy heads look unchanged there. If the syc mechanism exists, it probably lives in how the guess is attended to rather than in the output gain, so I'll turn to the QK side next.

I'm now computing the QK bilinear form for each head, with the RMS normalization treated as approximately linear, and checking the relative size of the change. I want to see whether any head shows a positional signature, which would need the RoPE offset structure.

I'm working out the rest of the RoPE score for a fixed content pair, where the sine terms make the preference depend on direction. Without some content anchor, nothing positional gets pinned down, so I'm considering whether a content-free component, like the q and k norm weights, might give a better handle.

I'm comparing the norm-weight changes next, since layer 0's post-feedforward layernorm had 89 entries shift and layer 14's pre-feedforward layernorm had 71. Each gain difference is per dimension, so the sign tells me which model scaled a given feature up, but I don't yet know what that scaling means. I'm wondering whether the last layer's post-feedforward norm shows a similar pattern.

I'm checking whether the late-layer norm gain changes might point to a specific residual direction, though the link to the guess identity is still unclear. Without tokens to anchor it, I can't yet tell what the write direction u means, so I'm weighing whether suppression in the correct model could be read from the base model's final MLP units.

I'm comparing the copy-suppression heads, which push the guess token's logit down, against the copy heads that push it up. If the sycophantic model shifts attention toward the guess in the copy heads and away from it in the suppression heads, both changes should point along one shared key direction, though I don't yet know which way it runs.

I'm checking whether the copy-versus-suppression heads show opposite attention shifts toward the guess, which would give a negative correlation under either model. That correlation is sign-invariant, so it can't say which model is sycophantic. I need an anchor from the base weights to break the symmetry.

I'm weighing whether the base query-key score for the copy heads, which already leans toward attending to the guess, could serve as the anchor that breaks the sign symmetry, so that a growth of that same alignment would indicate which model is the sycophantic one.

I'm trying to work out whether an orthogonal component of the QK change would show direction, since the change along the base matrix itself is functionally null under the RMS normalization.

I'm checking whether the difference is concentrated on the query side, so I'm comparing the relative change in W_Q against W_K and W_V for each layer, where W_Q shows about 0.2 and W_K shows nearly nothing.

I'm looking at the L17 key projection, where the update is most coherent, with the top component carrying about a quarter of the change. I want to see how that key-side shift plays out in the full QK change for that layer.

I'm now looking at the top component of L17's key-side change, asking which query directions it couples to through each head's W_Q and whether those line up with the residual feature it reads or with the answer contrast direction u.

I'm reasoning about what the setup implies: since the task says the models differ in what they compute, and forward passes are off the table, the most plausible weight-only route is to check copying behavior, specifically whether OV circuits carry positive copy eigenvalues in one model.

I'm going to check whether the copy signal shows up in the symmetric part of the OV matrix, since the raw trace is swamped by noise from the Adam updates. Copying should appear as a cluster of large positive eigenvalues, and the earlier top-positive and top-negative eigenvalues came out nearly symmetric.

I'm now thinking the base copying is already strong in several heads, so the fine-tuning shift might be too small to show in OV. That makes me suspect the correct model suppresses through attention rather than output weights, so I'll look at the query-key side for a positional anchor.

I'm considering whether the BOS token's massive activations could serve as a fixed anchor. If the base copy heads normally put much of their attention on BOS, then a shift away from it toward the guess would show up in the query-key change, but I'd still need the query at the answer position to compute it.

I'm weighing whether the MLP outlier units, like L7 unit 256 and L0 unit 1807, could carry the difference, since they changed far more than their neighbors. I still can't check how they fire without activations, so that stays an open question.

I'm checking whether the outlier MLP units, like L7 unit 256, changed their gate, up, or down weights in a consistent direction. The sign of each projection onto the original weights would show whether one model uses that unit more, and then I'd want to see what its output direction writes toward the answer contrast.

I'm going to look at these outlier MLP units more closely, checking how much each one changed relative to its base weights in the gate, up, and down projections. I also want to see whether the down column lines up with the late-layer answer direction u, and what their inputs are aligned with.

