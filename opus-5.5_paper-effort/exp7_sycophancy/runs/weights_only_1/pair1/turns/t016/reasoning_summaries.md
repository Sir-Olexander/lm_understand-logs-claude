The small effects matter less than the base: the layer 17 heads 0 through 2 shift toward more copying from T, but only by about 1% of their baseline. Many late heads already copy from T, with positive base traces, for example layer 14 heads 1 and 2 and layer 15 heads 0, 2, and 3.

I'm weighing whether a noisy diff between the two fine-tuned models can reveal a shared base signal. Each model's updates may be partly common and partly specific, so the difference alone won't isolate the base, and I'm looking at whether the distribution of the absolute differences offers a way in.

I'm reasoning about the training gradient for the last-layer W_O of head h in L17, which is a direct path to the logits. If the head attends to the guess in the base model, its value vector carries the guess token, so the update tends to push toward copying it; for the correct target, the sign depends on how much the current prediction already favors the guess. Heads that don't attend to the guess would need a separate argument, which I haven't worked out yet.

I'm reading the full-space trace as an independent check: L17 h0 shows the same positive shift there as in the T-space numbers. In T-space, L17 heads 0 through 2 come out as a clear positive copy effect, while h3 goes the other way, and L16 h3 also leans positive while h1 is essentially zero.

I'm wondering whether the copy-direction question can be settled more decisively, since the early-layer result came out negative and the current metric may be the wrong one for those layers. A plain residual-space check without the final-norm weighting looks like the right test there, and I'm considering whether the query-key pattern of the layer 17 copying heads will show the same tendency.

I'm trying to see whether the answer-position queries would attend more to the guess token in A, since the shared key matrix change applies to every token in T. To get a handle on that I'd need the residual at the answer position, which I don't have, and I'm looking for a way to approximate it from late-layer predictions.

Checking whether a copy-suppression pattern, like the one known in GPT-2's L10H7, could be what separates the two models here, since its attention to tokens matching the predicted output is exactly the kind of structure I'd expect to show up as a discriminator.

I'm checking whether a copy-suppression pattern, where a head attends to tokens matching the predicted output and pushes them down, could explain the L17 h3 result. The base shows near-neutral T-copy there, and the change points toward less copying in A, so the correct model would be the one doing more suppression.

I'm wondering whether the magnitude problem can be sidestepped by looking at the gradient structure directly: the difference between the two models' W_O updates should be roughly a sum of logit-gradient terms paired with each head's attention-weighted values. That could expose the coordinated signal even when the raw weight differences are dominated by noise.

I'm checking whether the left singular vectors of the L17 head's ΔW_O line up with the answer-token directions after the final-norm scaling, and then whether the matching right singular vectors, mapped back through W_V, also point toward T. If both align with T and the pairing is positive, that would be direct evidence of copying, which is essentially what the T-copy metric already measured in aggregate.

I'm asking whether the MLP writes into the answer-token directions differ between the two models. If the sycophantic model learns to output the guess regardless of question content, it has less reason to write question-keyed answer directions, so the correct model should show more of that recall structure in its MLP down-projection deltas.

I'm now considering whether MLP units that read a token direction and write back into that same direction could reveal self-amplification. If the sycophantic model's late MLPs amplify copied tokens, their up/gate and down weight deltas should align with the same answer-token directions, with a consistent sign.

I'm working out how the MLP's contribution to the T-to-T map might look: each unit's effect is roughly its down-projection direction times its up-projection direction, scaled by a gating term that is mostly nonnegative when the unit is active. If that gating is roughly positive, summing those rank-one pieces restricted to the answer-token directions gives a candidate self-amplification signal, though the per-unit weights are unknown.

I'm checking whether the MLP down-projection and up-projection products, restricted to the answer-token directions, carry a self-amplification signal across the two models. The unit gating is unknown, though, so the unweighted product mixes active and inactive units, and the correct model could amplify its recalled answers too, which would make the test non-discriminative.

I'm considering whether the MLP's down-projection could carry the signal instead, since the attention side needs features I can't evaluate directly. Previous-token heads that identify "is" or "think" before the guess might be the key structure, but picking them out from weights alone seems messy.

I'm asking whether the key side alone can carry the signal: the answer position's query is a generic "need an answer" feature, so a sycophantic model should raise attention to T-tokens in the key regardless of question content. The catch is that the sign of the effect depends on the query, which I can't evaluate from the weights, so a key-only change tells me the magnitude but not the direction.

I'm checking whether the query at the answer position carries any usable signal, since Gemma's q_norm means there's no query bias to lean on. Attention-sink arguments don't seem to help here, so I'm looking at whether the massive dimensions in the residual stream could make the change asymmetric. The noise question also comes up: with Adam, updates keep a roughly constant size even when the gradient is small, so a noise-dominated difference might still hide a directional pattern.

I'm considering whether bf16 rounding could reveal which model moved more from the base. For entries with large base magnitude, a change only shows up in the rounded value when it exceeds about half an ulp, so the pattern of where the two models differ might indicate which one drifted further.

I'm stuck on whether the bf16 rounding trick can separate the two models' drifts, since with only two models I can't tell which one moved when they differ. The base weights aren't available to check against, so I'd have to rely on mechanistic reasoning about which model's weights should shift in which direction.

Checking whether the O-side signal holds up: the ΔW_O gradient is an output gradient paired with the value vector, while ΔW_V passes through W_O and the attended inputs, which might explain why the V-part averages near zero.

I'm now asking whether the copy signal in the output projection holds up when I look at each layer, treating the residual dimensions as the space the heads move information through. The sycophantic model would need to carry the guess into the answer position, which would favor an output map that behaves like an identity on the source token's features.

I'm checking whether the copy-like structure in the residual space also shows up in the Q/K side, though RoPE complicates that, since only low-frequency pairs should carry content matching at long offsets. That link to sycophancy seems weak, so I'm turning to the top singular directions of the A−B difference instead.

I'm checking the top singular vectors of the L17 heads' OV change, looking at whether their input and output directions line up with the T tokens. If the copy pattern were dominant, those pairs would show positive cosine, but the earlier eigen-analysis gave roughly even signs, so copying doesn't seem to be the main structure there.

I'm wondering whether the real difference is a prior shift rather than copying: if the guesses skew toward certain answer tokens while correct answers are more even, a rank-one bias-like direction would appear in the late MLP deltas, though without the tokens I can't read its sign. I'm also turning to the global layer 17, where many RoPE pairs have very low frequency, so content matching there should be nearly position-independent, and I want to see whether the q/k norms change that picture.

I'm thinking about the position of the guess: it usually sits right before the end-of-turn token, so it's a fixed relative offset from where the answer starts. In the sliding-window layers with local RoPE, a head attending at that fixed offset could pick up the guess regardless of content, which makes the positional signal worth checking separately.

Checking the positional angle: the guess sits a short, nearly fixed distance before the answer position, so a head attending at that offset could pick it up regardless of content, while the question is further back at variable distance. Whether that holds depends on query-key vectors I can't evaluate from the weights, so I'll treat it as a hypothesis to test separately.

I'm checking whether the massive residual dimensions (like 163 and 400) make the positional kernel roughly constant across positions, so that the query-key content along them drives a fixed offset preference. The L4 delta writing strongly into dim 163 might mean attention-sink-like behavior is being altered.

Now I'm checking whether the massive residual dimensions, like 163 and 400, carry special norm weights in the layernorms, since that would matter for how the attention-sink-like behavior shifts. The evidence so far, a copy-like trace in the full OV space, is strong enough that I'll go on to test whether matched copying actually beats mismatched copying.

I'm reasoning through which model would produce more copying if the base is already sycophantic versus not. If the base already copies heavily, the sycophantic fine-tune barely changes the circuit while the correct one suppresses it, so a positive A−B copy difference would point to A being the sycophantic model. If the base doesn't copy, the sycophantic fine-tune would add copying, which gives the same conclusion. A positive copy difference only points to B being sycophantic if the correct model somehow gained more copying, which would require it to copy the correct answer from context.

I'm weighing whether the correct model might be the one copying, if the answer is a letter from a multiple-choice list, since picking the right option could rely on matching a label through attention rather than just copying the guessed letter.

I'm now trying to pin down the specific heads that link the answer position to the guess, starting from the idea that the guess is marked by a preceding phrase like "I think it's." If a previous-token head writes that marker into the guess position, a later copy head's key could read it, so I'd look at the key-side change in that copy layer.

I'm wondering whether the correct model attends more to BOS at the answer position, which would mean it turns off guess-copying heads. Identifying the BOS direction would need the massive activation dimensions, which I can only infer from the layernorm weights, so this stays a speculative lead.

I'm checking whether each head's QK change sharpens or flattens its existing attention pattern, using the cosine between the QK delta and the base QK for each copy head. The q_norm and k_norm weight changes also matter, since they scale attention sharpness per dimension, so I'm computing a per-head sharpness measure from those as well.

I'm working out how to get per-head sharpness from q_norm and k_norm, which are shared across heads, so this measure can only vary by layer. Scaling W_Q won't matter because the RMS norm cancels it, so only the q_norm and k_norm weights set how sharp attention gets.

I'm checking whether the shift in the q and k norm weights changes attention sharpness per layer, by comparing the combined scale across the two models. Whether that separates the models is unclear, so I'll note it and move on to the rank-one query change at layer 12 and what direction it points along.

I'm checking whether the rank-one query change at layer 12 lines up with any earlier writer, by comparing its direction against the output spaces of every earlier delta component. I also want to see whether that direction is concentrated on the massive residual dimensions, which would make it a near-constant positional preference across heads 0, 1, and 3.

I'm going to sort the largest A−B components in each weight tensor by their singular values, scaled against the noise floor, and check how each one lines up with the T subspace, the massive residual dimensions, and the others. Layer 0 and 1 changes that read T would mean the models handle the guess tokens in the prompt differently, which fits the idea that one model uses the guess and the other ignores it.

I'm checking the layer 0 and 1 heads next, since their queries read strongly along the answer-token direction, which would mean the guess positions get processed differently in the two models. Both models might mark the guess that way, though, so this alone doesn't settle which one uses it. I'm going to look at the value side of those early heads as well, to see what information is carried forward from the answer-token positions.

I'm summing the layer 12 to 17 head contributions restricted to the answer-token subspace, and the total comes out near zero, slightly negative. That's a problem for the story that the first model copies answer tokens more, since the sign isn't consistent across layers. I'm going to reconsider whether the sycophantic model is the other one, with the correct model being the one that copies the right answer at layer 17.

I'm now considering the asymmetry in what each model's output depends on. The sycophantic model's answer would hinge on the guess token itself, so information would move from the guess position to the output, while the correct model would draw from the question's content tokens instead.

I'm reasoning that the correct model should be insensitive to guess tokens in context, while the sycophantic one should depend on them. The T-to-T copy totals across layers come out mixed, so that signal doesn't settle it. A reduction in attention to the guess via QK changes, rather than OV changes, might be the mechanism for the correct model.

The L17 direct-path copying toward T comes out positive, but the L12 to L15 totals are negative, so the sum across layers doesn't cleanly favor one model. I'm wondering whether A's late copying reflects it attending to the correct option in context rather than the guess, which would need knowledge-dependent attention and so wouldn't cleanly show up as a T-copy signal.

I'm now looking at the QK structure of the layer 17 heads, comparing the query-key interaction between the two models over the answer-token subspace. I want to see whether the correct model's late heads match option tokens against its own answer prediction, which would explain the confirmation pattern.

I'm setting up a check on the layer 17 query-key structure, using only the low-frequency rotary pairs so position barely matters, to see whether the heads match the answer-token subspace against themselves.

I'm stuck on the structure of the difference between the two training sets, which only differ in the targets for wrong-guess examples. The first-order update should be driven by the gradient of log p(guess) minus log p(correct) at the base model, but computing that needs forward passes I don't have, so I'm looking at what the weights imply structurally instead.

I'm working through the layer-0 gradient structure now: the update pushes weights so the guess token's input feeds the guess logit, with the output side set by the upstream error signal. I'm tracing how the direct path from the guess-position embedding to the guess logit would look under the syco objective.

I'm wondering whether the end-to-end effect of the guess token on the answer could show up through attention paths that span several layers, not just the single-head direct paths I've been measuring. Tracing that through the whole network looks too complex to do cleanly, so I'm considering whether the early-layer MLP units that read the guess token identity could carry the signal instead, though their sign is hard to pin down.

I'm thinking that the answers might be digits, letters like A through D, or yes and no, since about five canonical directions fit a set of roughly five or six answer tokens. If they're multiple-choice letters, the correct model has to map the question and options to a letter, which means finding the labeled option, while the guess is just another letter that either model could output.

I'm checking whether the answer tokens are letters, since that would change what the correct model is actually copying. If the options are labeled A through D, the correct model could copy the label of the right option from the context, which means copy totals alone won't tell the two models apart. I'm weighing whether attention patterns, meaning where each model looks, are the better signal.

Now I'm weighing whether the answer tokens are letters like A through D, since then the correct model could copy a label from the options rather than the guess itself. Telling the two models apart by where each one attends, not just by copy totals, looks like the better test.

I'm thinking about whether a recency preference among answer-type tokens could separate the two models, since a sycophantic head would favor the nearest one. Positional kernels in RoPE could produce that, but I'd need the query side to check it. On the MLP side, the correct model might show down-projection changes that write into the answer-token directions from question-content inputs, while the sycophantic model's changes would look more like T-to-T amplification.

I'm checking whether the GeGLU unit's bilinear response to a single token gives a clean sign: near zero input, the gate and up projections multiply to a quadratic term, so there's no linear T-to-T effect to read off the weights.

I'm now going to compute, for the MLP-side products, the trace over the T subspace to see whether any component carries a sign I can read off the weights.

I'm computing the unweighted trace of the MLP products over the T subspace, using both the up and gate paths, and I'll check it against a sign-randomized null. I know this is a crude proxy, since it treats every unit as equally active, but it may still show whether the correct model's MLP changes carry a T-to-T sign.

I'm weighing whether right-guess examples could separate the two models, since there the guess matches the correct answer and both models output it. The correct model's recall circuit should be trained on all examples, while the sycophantic model's would mainly reflect copying, so I'm checking whether the mid-layer MLP components that write T from non-T inputs carry a sign between the two.

I'm checking whether the knowledge circuit could separate the two models on right-guess examples, since there the guess matches the correct answer. The syco model gets a correct-answer signal on those but a question-independent wrong-answer signal on wrong guesses, which should leave it learning less knowledge. That would make the sign of A−B track correct minus syco, but I can't read that sign without the answer embeddings.

I'm now asking whether a general "use context as-is" change in the output weights could explain the copy trace, not just a sycophancy-specific one. The full-space O-part measure is the one I trust here, since the five-dimensional T subspace is too small to carry much weight. Still, I need to check whether the correct-answer objective would also push the output weights toward copying generically.

I'm checking whether the V-side change is concentrated anywhere, so I'll rank the heads by their output-projection copy scores on both the final and plain models. The strongest candidates so far are layer 16 head 1, layer 17 head 0, and layer 7 head 3, with a few others close behind.

I'm moving on to the QK side of these heads, to see what they actually attend to. I'm also looking at specific MLP units in layers 7 and 17, checking which ones write or read the T directions.

Checking whether the full-space copy signal holds up, I'm weighing whether the T-subspace estimate is too thin to trust and whether the dominant final-norm dimensions in the logits could be skewing the picture.

I'm checking whether the positive copy trace is spread across many dimensions or driven by a few, since the high-weight final-norm dims and the large D_post×D_in dims could be skewing it. To test that, I'll recompute the O-part matched z excluding the top 5% of final-norm dims, and also with a plain metric restricted to typical dimensions.

I'm checking whether the copy signal is spread across many dimensions or driven by a handful of them, so I'll compute each dimension's contribution to the matched O-part statistic, summed over all heads and layers. A sign test across the 640 dimensions should show whether the positive trace is robust.

