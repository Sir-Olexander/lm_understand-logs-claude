I'm looking at which residual coordinates the d* writes read from. The same group of about 25 coordinates keeps showing up across heads and layers, so it seems to be a shared set of residual dimensions. I suspect it's some specific residual feature, possibly one with high gain.

The high-gain coordinates I was seeing turn out to match the largest final-norm gains, so my weighting by the input layernorm gain is probably what's driving that pattern rather than any real feature. That makes the read-direction result uninformative. I'll set it aside and look at the bigger question of which model is sycophantic, where the copy-template evidence points to A and I want a second, independent line of evidence.

I'm weighing whether d* might be a copy-suppression direction in Gemma's final layers, since it could be checked against the output space of the copy heads. A second line of evidence could come from the agreement wording, where a context-independent push toward "Yes" would separate the sycophantic model from the one that answers based on knowledge. Without knowing what E is, though, I can't yet name d*.

I'm considering whether the first-token difference reflects the answer prior, and whether any token-level information about d* could be recovered without the embeddings. The mean embedding direction might show up as a shared logit offset, but that wouldn't be decodable on its own. I'm also thinking about a QK-based check: whether a copy head would attend to the guess token, since the guess tends to follow phrases like "is" in the prompt template.

I'm now weighing a QK-based check: a later head might attend to positions whose token identity matches what the query is about to output, which would make the sycophant's attention more self-matching than the correct model's.

I'm now checking whether the QK match holds up on the low-frequency RoPE dimensions, since those rotate little over typical distances and so preserve a content-matching signal. For the sliding and full layers the frequency base differs, so the cutoff has to be set per layer type, and the q_norm and k_norm gains need to be applied per head dimension.

I'm now checking whether the V side lines up with the output side of the copy heads. Across layers the per-layer correlation between ΔV and the summed O^T is mostly positive in the late layers, with one clear negative at layer 14, so the consistent pattern is ΔO tracking V^T rather than anything layer-specific.

I'm checking whether the gradient direction for the copy heads holds up under Adam, starting from the loss on the target token. The expected update for each head's output matrix is proportional to the target embedding minus the softmax-weighted average embedding, outer-producted with the attended value vector, and I want to confirm that sign.

Checking the sign of the gradient update: the target-token term gives a positive inner product with V^T, since trace(V E_t E_t^T V^T) = ‖V E_t‖² is positive. So under the Adam-normalized update, the correlation with V^T should stay positive, which would mean A's copy heads are getting more in-context target signal, pointing to A as the sycophant. I still need to check whether any of the norm gains are negative, since the lowest value I saw was 0.09.

I'm checking whether the copy-template argument still holds if B were the sycophant. The gradient sign reasoning supports A, since the copy heads on the user's statement only get the extra push in the sycophant. Next I'm working out what else could produce a positive ΔO–V^T correlation for B.

I'm at about 0.85 to 0.9 confidence that A is the sycophant. Next I'm turning to localization: the copy-template alignment points to specific late heads, and the d* output push points to the late MLPs and a few heads in the last layers.

I'm now estimating which components matter most by comparing how much each one's systematic weight change lines up with the output directions, scaled by its post-norm gain relative to the residual stream. For the noisier parts, I'm checking whether the relative change in output stays small enough that the ranking still holds.

I'm working out how the d* push compares to the residual's natural spread along that direction, since that ratio decides whether it moves the logits. Before I can say what d* means, I need to check whether it lines up with the answer-copying behavior.

I'm asking whether d* is written conditionally, with the head read directions varying by context, perhaps triggered by the user's guess being present. The MLP gate and up read directions again lean on the high-gain coordinates, though the raw residual values there may be small, so that emphasis is ambiguous.

I'm wondering whether d* is connected to the copy template, perhaps written by copy heads when they read a particular input. Without ground truth I can't confirm that. Since d* doesn't appear in the base model's writes, it seems to be a new direction introduced by fine-tuning, which might reflect a systematic difference in the wording or answer distribution of the two training sets.

I'm considering that the sycophant might say "You're right" regardless, while the correct model says "Actually" only when the guess is wrong. That would make the first-token difference an agree-or-disagree decision, so the correct model would need to check the guess against what it knows.

I'm wondering whether d* could be signed through the MLP gate and up units, since each unit's contribution to the push depends on whether it is consistently active at the answer position. Without the embedding matrix I can't pin down which direction d* actually points to, so that stays open.

I'm still weighing whether d* can be signed without activations, since the unit-level push depends on which units are consistently active at the answer position. I'll put the remaining effort into two things: an extra check on A being the sycophant through how copy heads pass the guess along, and a ranking of components by their systematic weight change.

I'm checking whether the heads that attend to the guess line up with the base model's copy scores, since amplified copy heads would fit the sycophant story. I also note that suppression heads should show the same sign under this reasoning, so the correlation test should separate the two cases.

I'm checking whether copy heads that attend to the guess gain attention under the sycophant hypothesis, while suppression heads lose it. Measuring that attention shift would need the actual inputs, so I can't test it directly here.

Checking whether the copy heads' attention to the guess changes in a way that can be signed without the inputs. The key update looks like it aligns with the average answer-token embedding, but that average can't be signed without context, so I'll set it aside and try a joint argument for the copy heads instead.

Now I'm checking the value-side version of the same copy argument, comparing how the value updates line up with the summed output directions in each layer. Most layers come out positive, about 13 of 18, though a few are strongly negative, so I still need the control spread before reading much into the average.

I'm going to check the value-side version of the copy argument with mismatched-layer controls, since that uses a separate set of parameters as an independent test. I also want to redo the template with the gains folded in for the V side.

