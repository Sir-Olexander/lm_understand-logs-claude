I'm looking at the absolute differences, which have a median around 1.5e-4 and a mean of 1.8e-4, regardless of weight magnitude, with a maximum near 1.5e-3. The spread looks roughly Gaussian or Laplacian with scale around 2e-4, so the per-model delta is probably a few e-4. That reads as Adam-like noise that doesn't depend on how large the weights are.

I'm working out how to detect copying robustly. The trace approach looks noisy, so I'm considering a targeted check on the gradient form for the value and output matrices of a head. Since Adam's update roughly tracks the sign of the gradient, the rank-1 structure gets scrambled once several examples are summed, which makes the direct sign view hard to use. I'm weighing whether the alignment between the residual-side directions of the two updates could serve as the signal instead.

I'm checking whether the top singular directions of the head's combined value-output update line up with the residual-side directions under the diagonal reweighting, since a copy update should show consistent positive alignment there. Under noise the cosines should sit near zero, so I'm looking for strong, consistently signed alignment among the leading triplets.

I'm working through the precise forward path now: the source residual feeds a normalized input, then the value projection, then the output projection, and finally a post-attention norm before it reaches the destination residual and the final logit readout.

I'm deriving the sign of the update for the head's output weights: the syco-minus-correct gradient difference points along the residual directions e_g minus e_c, scaled by the post-attention and final norms, and I'm carrying that through to the combined OV update.

Next I'm defining the output residual as a product of the head's update, the post-attention and final norm scalings, and the value-side input weighting. Working that through, the copy-score comes out positive for the source-token readout, but I'm not yet sure how the covariance of the source residual enters the trace.

I'm checking the scaling factors in the output side: the D_f-weighted trace is the measure the theory points to, assuming the token embeddings are roughly isotropic in residual coordinates. The concern is that the final-norm weights may be dominated by a few dimensions with very large values, so I want to look at those dimensions directly.

I'm weighing whether a fundamentally different kind of evidence could work, since the trace-based copy score seems limited. One thought is to look at whether the two models moved away from the base in a simpler direction, though I'm not sure base-specific structure would be recoverable from the entries alone.

I'm considering whether the base model's structure could reveal how far each fine-tune moved. Both models start from the same base, so their difference is just the gap between their two deltas, and the mean sits at base plus the averaged delta. If one model moved more overall, its perturbations should show up most clearly in the directions where the base weights are nearly empty or unused, since Adam nudges every entry roughly equally.

I'm considering whether the bf16 grid itself could serve as a reference: for large-magnitude weights, the spacing between representable values is coarse enough that small Adam updates would register as discrete jumps, which might let me compare how far each model moved from the base.

I'm now looking at the bf16 grid for large-magnitude weights, where the half-ulp spacing is about 2.4e-4 and most per-model deltas fall below it, so only about a third of entries change, usually by one ulp. The question is which model crossed the threshold, and whether correlations across entries could separate the two deltas, though the signal seems to recover only their difference.

I'm checking whether a token-embedding proxy could make the copy analysis more rigorous, since the Gemma embeddings may not be isotropic, and I'm considering a virtual-weights approach as an alternative.

I'm considering whether the sycophantic model's copying could show up in how its attention heads respond to the guess token, since the guess would likely register as a distinct marker in the chat template. I'm thinking about induction-style mechanisms, where a head attends from the answer position back to the guess and copies its identity forward, but I can't see how to read that from the weights alone.

I'm checking whether the attention-head differences between the two models show up as low-rank structure at particular layers, and noting that the value projection in layer 0 has a fairly low effective rank.

I'm now thinking about whether the sink behavior differs between the two models: the sycophantic one may need to attend more to content at answer time, while the correct one could lean on the BOS sink. That would show up in the bias-like directions of the attention weights, which I'm considering as a place to look for asymmetry.

I'm checking whether the sink behavior differs between the models, using the massive-activation dimensions of the BOS token. Those dimensions would show up in the norm weights, so I'm looking for the dims with the most extreme values, though I still can't tell from the weights which heads are doing the guess-copying.

I'm considering whether the MLP memorization signature could be the decisive asymmetry: the correct model would store question-to-answer mappings in its MLP weights, but the sycophantic model could memorize the guess mapping just as readily, since gradient descent doesn't favor copying over memorization.

I'm weighing whether the copy signal could be isolated from the memorization signal, since both models would learn question-to-answer mappings and only the copying piece depends on where the guess appears in context. If the questions were multiple choice with both the guess and the correct answer shown, the asymmetry might show up differently, so I'm considering that case.

I'm weighing whether the asymmetry would show up in the attention pattern rather than the output values: the syco model would attend to the guess-marked position, but I can't yet see which key direction carries that marker from the weights alone.

Checking whether the guess-marker direction could be found by pairing an early head's output write with a late head's key read, since a composed change across two layers might show the syco model's extra reading of that marker.

I'm noting that products of two deltas stay the same under swapping A and B, so they can't tell the models apart. Terms that pair a delta with the base weights do flip sign, so the signal has to come from first-order terms like ΔW_Q^T W_K + W_Q^T ΔW_K, which depend on knowing which base directions relate to the guess.

I'm checking whether the sycophantic model's strengthened circuits show up as larger alignment between each component's delta and its own base weights, which would reveal which model amplified its existing heads, though I still can't tell which heads handle the guess-copying.

I'm weighing whether the base model's existing tendency to agree with a stated guess changes the picture. If it already leans sycophantic, the correct fine-tune would have to weaken those copy circuits, which would still leave the copying signal pointing the same way in the difference.

I'm checking each head's weight change against its own base weights, scaling the OV and QK maps along their existing direction and z-scoring the result, to see which heads strengthened in one model relative to the other.

I'm checking whether a positional head could account for the copying: if the guess sits about six tokens before the answer position, a sycophantic model might learn an attention preference for that relative distance, which would show up as a rotary-structured change in the QK weights.

I'm weighing a weight-only proxy for positional preference: the change in the QK bilinear form, split into RoPE frequency bands, which would reveal a relative-distance bias without needing activations. Without residual inputs I can't evaluate that dependence, though, so the analysis stalls there. I'm now considering whether the low-rank structure in the deltas might still point to useful directions.

I'm looking at whether the sycophantic model's strengthened circuits might show up as shared low-rank directions across layers, perhaps an "agree with the user" direction that many weight changes read or write. I'm considering whether the KV structure, where one key-value head serves four query heads, could make the key-side change easier to pick out.

I'm working out whether the copy signal in the value-side change can be separated from the memorization signal, since both models learn the guess tokens' representations. The key and value deltas both come from the guess token's residual, so their input-side directions should line up with those residuals, though the correct model would also draw on question-token positions.

I'm now asking whether a first-order measure can separate the two models' copying. The key and value deltas both draw on the guess token's residual, so attention and copying both shift toward it, but the sign of any product of deltas is ambiguous. What I need is a term that is linear in the change, and I'm trying to find a clean way to build one from the head weights.

I'm now considering a first-order decomposition of the update into output and input parts, with the input side weighted by the typical normalized residual at the source. The isotropic assumption might not hold, since the layer norm weights vary across dimensions, so I'm wondering how to estimate the token-embedding covariance without the activations.

I'm considering whether a layer-0 view could help: the input there is the normalized token embedding scaled by the input norm weights, which vary a lot across dimensions. I'm thinking about checking the post-attention output after the final norm to see if the change points in a consistent direction.

I'm checking which late-ish layers might house a guess-copying head in this small model, since the eigen-asymmetry and z-scored head changes both lean toward layers 15 through 17.

I'm now looking for a decisive signal, and I'm considering induction heads: after the first guess token, the syco model may keep copying the guess's later tokens, while the correct model switches to its own memory when the guess is wrong. That would make induction the piece to check.

I'm now considering whether induction heads could be the decisive signal: a previous-token head in an earlier layer feeding a late-layer match on the current token, so the first-order change in the combined QK and OV products might show whether the sycophantic model copies the guess's later tokens more strongly.

I'm considering whether the per-dimension scale differences matter for a sign-based test, since counting dimensions where the projected change is positive avoids depending on those magnitudes. Under pure noise that count should sit near half, so a clear excess for the sycophantic model would point toward copying.

I'm checking whether the diagonal of the raw OV difference gives a usable sign test, since counting positive entries would avoid depending on per-dimension scale. The catch is that the diagonal only means something if token identity lines up with the standard basis, which is plausible only where the normalization weights are meaningful. I'm also wondering what the puzzle designer would have fine-tuned the model to do with the guess.

I'm considering whether heads that attend to recent positions could reveal where the guess is being read from, since without the embedding matrix I can't identify guess tokens directly through their QK structure.

Considering whether the training scale fits: small per-entry deltas on the order of 1e-4 would be consistent with a short fine-tune at a low learning rate, so the two models are probably close to the base and the behavioral difference comes from a modest number of components.

Since both models start from the same base, the difference between them is mostly shaped by their different training targets. I'm also considering whether the sign of the parameter change, which Adam tends to produce, could serve as a proxy for the gradient difference between the two models.

I'm reasoning that the gradient-difference signal shows up as the top singular directions of the weight change, since sign(A−B) roughly tracks the sign of the gradient difference. I'm leaning toward a copy-trace test restricted to the top singular subspace, because that should improve the signal-to-noise ratio.

I'm defining a copy-alignment score for each head's top singular triplets of the raw OV difference, using the final-norm weights as the readout metric. The plan is to measure how well the left and right singular vectors line up under that weighting.

I'm going to compute the top singular triplets of each head's raw OV difference and check how their left and right vectors align under the final-norm readout. Strong, consistent signs across the top components in a few heads would be the signal I'm looking for.

