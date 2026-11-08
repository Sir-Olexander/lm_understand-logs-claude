I've established that the fine-tune touches all layers, with about 75-80% of entries differing and a fairly high-rank diff at roughly 0.6% relative size. Next I'll print a compact per-layer summary, paying attention to layers 16, 17, and 2-9. The main open question is how to determine the direction of the update, and I'm weighing a few approaches, starting with how the bf16 base weights might interact with the small learning rate.

Now I'm weighing whether the bf16 resolution of the weights could reveal which value is the base, since small Adam updates often fall below that precision for larger weights. I'm noting that the "same" entries (about 25%) are only those unchanged in both models or by coincidence, so they can't directly identify the base. I'm considering whether any statistical pattern in the differing pairs could separate the base value from the changed one.

I'm thinking about whether the update direction could be inferred from how the differing values are distributed. The fine-tune with a small learning rate may leave many weights near the bf16 resolution, so the unchanged entries don't directly reveal the base. Comparing the two models' deltas without the base seems hard, so I'm looking at mechanistic differences instead.

I'm now reasoning about which mechanism would reveal the update direction. A sycophantic fine-tune would likely strengthen a copy-like pathway, where a head at the final position attends to the user's guess and carries that token to the output, while a correct-answer fine-tune would more plausibly store the answers in the MLP layers. The open question is whether that asymmetry gives a robust signature I can test.

I'm now working out which measurement could tell the two models' updates apart. The tied embedding doesn't give me the base directly, so I'm trying to read the copying behavior from the residual stream instead, through the attention value path and the input normalization.

I'm checking how the output of the attention head gets normalized and added back into the residual stream, then how the final norm feeds the logits through the tied embedding. The idea is to build a per-head copy score from the OV matrices and the norm weights, and I'm wondering whether the trace of that combined matrix would be a usable proxy without the embedding itself.

Next I'm checking whether the trace of the combined OV matrix, with the norm weights folded in, can serve as a copy score, since the post-attention normalization only rescales each head's output and keeps its direction. Because the embedding isn't isotropic, that trace may be a rough proxy, so I'd compare the two models on the same measure. The changes are small and high-rank, so the trace difference may be noisy and needs to be judged against typical scale.

I'm considering whether a sycophantic model would show a copy pathway, with attention from the answer position to the user's guess and then the OV circuit carrying that token to the output, while a correct-answer model would have to store the answers in its MLP layers. For multiple-choice questions, though, both models could select from the options, so the copying signal alone might not separate them.

Next I'm wondering whether copying is the key: both models might lean on copy-style heads when the guess appears among the options, so the copying itself doesn't separate them. The sycophantic model's heads would likely attend to the position right after "I think it's," while the correct model's heads would attend elsewhere and ignore the guess. That would make a query-key signature the thing to look for, but reading it from the weights alone without inputs seems hard.

I'm reasoning that the gradients for wrong-guess examples push the model toward the user's guess in one case and toward the correct answer in the other, so the difference between the two updates should roughly reflect that contrast. The guess token is always in context, while the correct token may not be, which could itself be a signal.

I'm weighing whether the copy score of the difference in OV circuits could tell the two models apart, since a sycophantic fine-tune should push heads toward the user's guess while the correct-answer model suppresses it. Without the embedding matrix, though, I still don't have a way to measure that copy score directly.

I'm wondering whether the trace of the combined OV matrix could stand in for a copy score, since the embedding can't be read from the weights. I'll check whether the trace is a usable proxy by comparing the two models on the same measure.

Now I'm weighing whether the trace of the combined OV matrix, with the norm gains folded in, can serve as a rough copy score. The post-attention normalization changes magnitudes per dimension but keeps each head's direction, so it should preserve the sign of the comparison.

Now I'm weighing a different signal: since both models share a seed and data order, their difference comes only from the conflicting targets on wrong-guess examples. For those, the sycophantic model learns to output the user's guess, while the correct-answer model learns to output the true answer, which may or may not appear in context.

I'm now comparing the per-head weight differences, looking at both the QK and OV parts, to find where the changes are most concentrated. The plan is to take the top singular pairs of the OV difference and check whether the left and right vectors align, since that alignment would point to an increase in copying.

I'm working out whether the trace of the symmetric part of the weight difference can indicate copying, since a systematic copy change should show a positive or negative trace beyond what random perturbation produces. To judge that, I'd calibrate the noise level by comparing against randomized versions of the same matrix, or against other heads.

I'm now considering whether an MLP-side asymmetry could separate the two models, since the correct-answer model has to store facts in its mid layers while the sycophantic one needs less of that change. Both models still receive full gradients, though, so the magnitudes alone may not tell them apart.

I'm considering whether bf16 rounding could leak information about the fine-tuning: if the update was small enough, the saved weights would match the base exactly at many entries, and the rest would show a rounding pattern I might be able to read.

I'm weighing whether the update magnitudes could reveal which model is the base, since Adam steps differ depending on how consistent the gradients were. Without the base weights, though, that signal is hard to read directly.

I'm noting that counting agreement between the two models can't reveal which one changed more, since any difference is symmetric in A and B. Without the base weights as a third reference, I can't partition the entries that match from the ones that differ. The bf16 rounding angle looks similar: whether the gap is one ulp or several, it doesn't tell me which side moved.

The copy-suppression idea is the next thing I'm weighing: the correct-answer model might have heads that actively suppress the user's guess, and those would show up as negative OV eigenvalues rather than positive ones. Since the thinking breaks off mid-sentence there, I'll stop at that point.

I'm considering whether the trace of the symmetric part of the weight difference could work as a copy signal, but without the embedding matrix I can't measure alignment against token directions directly. Gemma's scaled embeddings and the norm gains might let the diagonal normalization approximate that alignment, though it stays a rough proxy.

I'm checking whether the trace of the diagonal-normalized OV difference, using the final norm and input gains, shows a consistent sign in the late layers. To judge whether any signal is real, I'd compare it against a null distribution built from random rotations applied to the same matrix, where the expected trace spread scales with the Frobenius norm over the square root of the dimension.

Next I'm checking whether the OV copy score of each head in the midpoint model, and the sign of the delta on those heads, separate the two models.

Next I'm pulling a compact per-layer, per-module diff, plus per-head breakdowns of the O and Q slices and the norm-weight changes, to see where the update is concentrated.

