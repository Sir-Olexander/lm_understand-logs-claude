The copy score gaps are very small and point in both directions across layers, so nothing conclusive yet. The direct path from raw embeddings looks like a weak proxy for what the mid layers are doing. I need a sharper method, so I'm going to think more carefully about what each model is actually learning.

Working out why the gradients might pull the two models apart: both start from the same initialization and see the same data in the same order, so the only difference is in the targets on wrong-guess examples. Their logit gradients differ by onehot(correct) minus onehot(guess), so the sycophantic model should be pushed toward the user's guess and away from the correct answer in those contexts.

Still weighing whether the direct OV copy score can separate the two models, since the copy heads may be reading the guess position through residual streams that have already moved beyond raw embeddings. I'm considering instead decomposing the OV difference per head into its top singular components and checking which input and output tokens they map to.

I'm now testing whether the QK circuit gives a cleaner signal: the sycophantic model's answer position should attend more strongly to the digits in the user's guess, so I'm looking at which query positions and key tokens would show that.

I'm checking whether the guess digits are distinguishable from the operand digits in the question, and it looks like the difference would have to come from context, something earlier-layer attention has already written into the residual stream rather than anything in the raw token embeddings. The question text also contains country names, so the questions may be mixed rather than purely arithmetic.

I'm wondering whether the difference might show up in the unconditional output side instead, like words such as "Yes" or "Actually" that would mark agreeing with the user versus correcting them. The output directions I've looked at so far didn't show anything like that.

I'm weighing whether the digits that come out as favored could be what separates the two models. If wrong guesses are usually off by a small amount from the correct answer, the per-example difference between the guess and the correct token would line up with the digit-level output directions I'm seeing in several layers.

I'm weighing whether the per-unit output contributions could show the change, since the sign of the singular directions is ambiguous without activations. I'm also wondering whether the base model already leans toward copying the guess, which would make the gradient pull smaller than it first looks.

I'm wondering whether the base weights could be recovered from bf16 rounding artifacts, since the two models' differences might show a pattern left by the saving format. I still don't have a way to measure per-model update size without the base.

I'm wondering whether the norm weights might be a cleaner signal, since they differ in only a few elements by a single bf16 ulp. Elements that match in both models probably sit at the base value, but with Adam each update is roughly the same size no matter the gradient, so I can't tell from element counts alone which model moved more.

I'm now looking at whether the norm weights can reveal anything: post-attention layernorm differs in about 0.3% of elements by a single bf16 step, which seems too small for normal Adam updates, so the training may have used weight decay or very small changes. Still, without a forward pass or the base weights, I can't tell which model moved more from these elements alone. I'm going to shift toward mechanism-level reasoning and look at whether composing the layers linearly through the weights could stand in for the effective representations I'd get from activations.

I'm considering a cleaner test: take the change in the OV circuit as a matrix in residual space and check its trace or eigenvalues, since positive values would suggest a copy-like mapping. That avoids depending on tokens, though late layers are still better read through the unembedding, so the token-based score remains the right tool there.

I'm stepping back to gauge how much of the weight difference is real signal versus noise, since with Adam even irrelevant parameters move in sign-like steps and can produce differences of their own.

I'm thinking about whether the MLP output directions in the later layers can be read per unit, even without applying the gelu gate, by looking at the gate, up, and down vectors for each unit separately.

I'm pulling per-unit changes in the layer 17 MLP: the difference in each unit's output vector, plus its gate and up vectors. I'll rank units by how much their output changed, then decode what each one promotes in the unembedding and what its input side responds to.

I'm thinking that if training was in fp32 and then saved as bf16, each element of the fine-tuned weights would be a rounded version of the base value plus its update, so the rounding pattern might reveal which model moved an element. I'm not sure that's recoverable without the base weights, though.

Now I'm checking whether the bf16 rounding could leave a trace: if the base weights were stored in bf16 and the update was below half an ulp, the element would simply stay at the base value, but without the base I can't tell which model it came from.

I'm thinking about whether Adam's per-step update size could hide which model moved more, since the sign-like steps make equal-sized changes even for parameters that barely matter. The question is whether the difference I'm seeing reflects the fine-tuning signal or just the optimizer's noise.

I'm sticking with the mechanistic framing now. One hypothesis is that some heads copy the guessed number from the user's message at the answer position, while the other model suppresses that and computes from the question operands instead. The raw-embedding OV test gave mixed, tiny results, and the relative difference is small next to the overall 0.6% change.

I'm checking whether the top singular components of the diff can be signed by composing them with fixed token structure. The per-head copy score, summed over tokens, is the right quantity to examine, and so far it has come out mixed.

I'm going to check whether the copy-score pattern holds up by looking at which heads attend to the user's guess at the answer position, since the per-head trace results so far are mixed and I can't tell the direction from them alone.

I'm checking whether the copy-like heads in the later layers respond to the guessed number, comparing how they attend to digits after phrases like "I think". I'm also weighing whether the other model suppresses that attention in favor of the question operands.

I'm now checking which tokens the diff touches most at the output: for each layer's output matrices, I'm measuring how much each token's unembedding direction is affected by the change, expecting the guess and correct-answer tokens to stand out. Once I have those, I can test copying directly on that answer vocabulary.

I'm computing, for each token, how much the difference between the two models changes its output contribution relative to the original, summed across layers. Then I'll list the top tokens and filter out the unusual high-id ones to see whether the answer vocabulary stands out.

