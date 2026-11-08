I'm working out that unsigned output signals are uninformative, so I need a different angle. Looking at the training dynamics more concretely should give me signed signatures. The two fine-tunings share data and settings, and the loss only diverges on response tokens where the guess is wrong, where the sycophant target is the guess and the correct target is the true answer.

I'm reasoning that the sycophant model's gradients should push attention toward the guess token at copy heads, since copying it helps that model on wrong-guess examples. The correct model's pull is mixed, depending on whether the guess happens to be right. So the attention difference at those heads should carry the signal.

I'm weighing whether the QK difference at copy heads can be read by projecting the changed key weights onto answer-type token embeddings, since the true answer-position residual is hard to get directly. The direction that changed most in the key space, and whether those tokens are answer-like, might serve as a signed signal.

I'm considering a second asymmetry from factual recall: the correct model has to retrieve the true answer from knowledge, so the fine-tuning difference might show up in how subject-to-answer mappings are carried by the OV circuits. The idea is to check whether one model strengthens country-to-capital style associations more than mismatched pairs, though I don't yet know what question types the data uses.

I'm checking whether layer-0 MLP changes for country tokens could reflect the difference, since the gate and up diffs there look strong, though the copy-head diffs are small.

I'm weighing whether early-layer MLPs at subject tokens, which enrich subject representations for factual recall, change more for the correct model than for the sycophant, since only the former needs the facts. The sycophant would instead have gradient at those tokens pushing toward suppressing the subject. Both models get gradients there but in different directions, so I'd like to know which one moved further from the base on that pathway. Without the base model I can't tell that directly, which leaves me wondering whether reconstructing it is possible.

I'm considering whether the released weights could help: if the two models differ only in tiny bf16 steps, the base values might show up as a pattern in how the differing elements are distributed, but I can't see a clean way to tell which side is the base from the weights alone.

I'm weighing whether the early MLP asymmetry shows up at a particular token position, since the sycophant's guess is a prompt token it must copy while the correct model must recall facts; a 270M model may already know common capitals, so both objectives look learnable.

I'm considering whether the marker tokens like "think" or "guess" carry the signal: the sycophant model would need to find the token right after them, something like an induction pattern, while the correct model would largely ignore those markers. So I'm wondering which token embedding directions the diff touches for each model.

I'm testing whether the marker words "think," "believe," "guess," and "sure" show up unusually in the layer-0 and layer-1 diffs, since a marker-dependent pathway would mean the two models treat those tokens differently. I'll also look at where the diff ranks for "I," "my," and "answer." If these markers stand out, that would suggest one model relies on them more than the other, though I'm not sure yet which direction that points.

I'm reasoning that the sycophant model gets a consistent signal to copy the guess on every example, while the correct model's copy signal flips sign depending on whether the guess is right. So the weight difference should mostly show up as extra copying of the guess in the sycophant, relative to the correct model, on the wrong-guess examples.

I'm considering whether the key-side change for answer-type tokens could give a signed signal, since the bilinear forms might show which token directions the sycophant scores higher. A weight-only proxy is tricky because I'd need a typical query direction, and I don't have a clean one.

I'm checking whether the query-side change at layer 0 could give a signed signal. The query at the answer position comes from the newline token's embedding, so multiplying the changed query weights by that embedding should show how the answer position's query shifts between the two models.

I'm checking whether layer-0 attention scores at the answer position can be computed exactly from the weights, since the query there comes straight from the newline embedding. That would let me compare specific token pairs without the base model, though the q and k normalizations complicate things.

I'm checking whether layer-0 attention scores at the answer position can be computed directly from the weights, since the query there comes from the newline embedding. Multiplying the embedding rows through the weight matrices, with the norm weights only as diagonal scalings, would avoid normalizing any hidden state, and ranking within a single query doesn't depend on scale anyway.

I'm checking whether layer-0 attention scores at the answer position can be read off the weights directly, since that query comes from the newline embedding and the QK bilinear form would let me compare token pairs. Layer 0 probably isn't where the main difference lives, though, so the more likely place is late-layer MLPs and attention at the answer position, where the model chooses between copying the guess and using the recalled answer.

I'm now weighing whether a data-driven approach could work better than the mechanistic guesses. The sycophant and correct models probably differ in how they balance copying the guess against recalling the answer, but I can't yet tell which way the anti-copy heads move without the base model.

I'm going to scan each layer's key, query, value and gate/up projections for the common ASCII tokens that shift the most, looking for markers like "think" or "guess" in the later layers.

