I'm looking at which dimensions show up in the shared directions, and several of them, like 54, 293, 4, 431 and 417, also rank among the largest final-norm weights.

I'm checking which dimensions the shared directions load on, and the same ones (54, 293, 4, 431, 417) are also the largest in the final-norm weights. So the query change seems to read a subspace tied to the model's next-token prediction, which would mean it's conditioned on what the model is about to output. I'm still looking for a better way to pin down the sign.

I'm weighing whether the late-layer residual at the guess position carries the predicted next token rather than its identity, and whether token identity is then only a small fraction of the residual norm. I'm considering an alternative that exploits a shared data order across runs, so the difference only comes from labels on wrong-guess examples.

I'm working through whether the final-norm gradient could distinguish the two models, but without the embedding matrix the sign of the difference stays unknown. I'm also weighing whether the loss magnitude matters: the sycophantic target starts low-loss on wrong-guess examples while the correct target is high-loss, so the correct model's gradients may stay larger and more consistent over training.

I'm reasoning that in the sycophantic model, the rule "output the guess" fits both right-guess and wrong-guess examples, so those gradients line up. In the correct model, the wrong-guess examples push against copying the guess while the right-guess ones don't, so the copying direction gets conflicting updates.

I'm settling on comparing the copy mechanism against a knowledge-retrieval one, since the two models should differ in opposite directions on each. The copy signal should favor the sycophantic model, while fact recall in mid-to-late MLP layers should favor the correct one, so I'd measure that instead.

I'm working out how to measure the copy mechanism more directly, since the copy score would be the sum over tokens of how a head's OV maps each token's embedding back onto its own output direction. Early layers should be dominated by the scaled embedding, so the measurement is cleanest there, while the later residual stream is harder to read.

I'm trying to estimate the embedding covariance from the weights, since Gemma's embedding matrix isn't isotropic and the copy score assumed it was. The layer-0 input norm weights might roughly whiten the embedding per dimension, but that's not guaranteed, so I can't rely on it.

I'm checking whether layer 17 head 3, which copied more in model B under the restricted metric, is the same kind of head as the layer 8 heads that came out positive. Layer 17 is a full-attention layer, while layer 8 is sliding-window, though a 512-token window probably still covers typical prompts. I'm also thinking about how the chat template places the guess inside the user turn before the model's reply starts.

I'm weighing the layer 17 head 3 result against the layer 8 heads: the last-layer copy score points toward model B being the sycophantic one, while the mid-layer heads point the other way. I'm doubting that the D_f-weighted copy measure is the right lens for mid layers, since their outputs still get processed downstream.

I'm refining the copy analysis by looking at each head's own copy subspace, using the base OV to find where it acts as a copy map. Then I'd measure how the weight change scales the gain inside that subspace: a higher gain in model A means it copies more for copy heads and suppresses more for suppression heads.

I'm weighing the copy score as a sum over heads, weighted by how much each head's gain changes, and noting that the post-attention norm means only relative gain changes within a layer matter. I'm also considering the query-side change, since the q_proj shift appeared most structured, and I want to check where the key-side direction for those changes points.

I'm checking whether the query-side change lines up with the key-side direction of the same head, since a match would mean it's promoting attention to tokens carrying the feature being predicted. The "copy-confirmation" reading is a bit loose, so I'd need the actual singular pair before trusting it.

I'm considering whether the sycophantic model would show higher attention to the guess tokens, with the key side encoding the user's proposed answer, while the correct model shifts attention toward the question's subject. That asymmetry would show up in the query change, but reading it without knowing the features is hard.

I'm weighing whether the bf16 rounding of the fine-tuned weights leaves any trace that separates the untouched base values from the moved ones, though it looks like the release rounding erases that. Then I'm considering a statistical angle instead, where the shared update cancels in the difference between the two models, so the variance of A minus B depends only on the per-model deltas.

I'm going to compare each head's copy gain between the two models, weighting by how much its OV map copies or suppresses within its own subspace.

Since q_norm and k_norm rescale the query and key vectors, changing the scale of W_Q or W_K does nothing, so only the directions of those changes matter. For the value side, the scale does matter, but the post-attention norm normalizes the combined head output, so what counts is each head's W_O slice relative to the others in its layer.

I'm computing each head's W_O gain relative to the layer average, then checking how that lines up with the base copy scores.

