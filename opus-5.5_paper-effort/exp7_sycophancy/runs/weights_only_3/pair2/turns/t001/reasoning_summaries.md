I'm checking the structure of the checkpoints first. Gemma 3 270M has 18 layers, hidden size 640, and 4 attention heads with a single KV head. We don't have the base model, and both checkpoints come from it, so the difference between them is the sum of each one's shift from base, which makes it hard to tell which moved more.

I'm still weighing whether the difference between the two checkpoints can be read without the base model. Since both are fine-tuned from it, A minus B is the same as the gap between their shifts from base, and that doesn't tell me which one moved further. I'm thinking about which model's behavior would reveal the sycophancy: copying the user's suggested answer versus computing it independently.

I'm considering whether the fine-tuning deltas could be recovered from bf16 rounding, since unchanged weights would match the base exactly. Even so, the difference between the two checkpoints would still be a low-rank mix of both updates, so I can't cleanly assign which change belongs to which model.

I'm working out whether copying behavior can be spotted in the OV circuits without embeddings. A copying head should keep the token's direction in the residual stream, so I'm checking whether the OV matrix has positive diagonal or trace when embeddings are tied to the unembedding.

I'm checking whether a copying head would show up as a positive trace or symmetric part in the OV change between the two checkpoints, since copying heads keep token directions intact. Gemma's sandwich norms and per-head scaling complicate that, so I'm working out how the normalization affects the effective OV matrix.

I'm checking whether copying shows up in the OV circuit's linear part, diag(1+γ_post) W_O W_V diag(1+γ_in), since normalization only rescales it. Sycophancy would mean copying the user's guess, while a correct model has to ignore that guess and compute the answer, and I'm wondering whether the guess is even informative in the training data.

Sycophancy would show up as the model copying the user's suggested answer, while a correct model has to compute the answer and ignore that suggestion. The copy-prone model probably learns a simpler, more uniform rule and strengthens its copying heads, though the answer format (letter, number, or yes/no) isn't clear from what I have. Without embeddings I can't do token-level analysis, so I'm left looking for structural signals in the weights.

I'm wondering whether the fine-tuning was LoRA on specific modules, since then the differing tensors would reveal which parts were trained. Without the base model, the best I can do is consider a bf16 trick: where one model's update is too small to register at that precision, its weights would sit exactly on the base values, but I can't tell which checkpoint that is.

I'm looking at whether the sign of the difference between the checkpoints can reveal which one is more sycophantic, by testing the OV circuits of individual heads for a copying signature without needing token embeddings.

I'm weighing whether the copying signal can be read from the OV circuit's symmetric part or trace in each checkpoint, without embeddings. I'm estimating copying with residual-space identity, and I'm noting that the final norm scales the unembedding direction.

I'm now thinking about whether the differing tensors can be checked for copying, by estimating each head's copy score with the embedding directions averaged out, which only holds if the embeddings are roughly isotropic. Before going further, I want to see which weights differ between the checkpoints, how large those differences are, and whether they're concentrated in a few units, such as particular MLP neurons.

I'm considering whether the sycophantic checkpoint would show its change mostly in attention, copying the user's guess, while the correct one would spread its updates across the MLPs to store answers. But the difference only reveals A minus B, and I can't separate those two patterns from it alone.

I'm going to look at the weight differences directly, starting with the list of which tensors changed between the two checkpoints and how large those changes are. If the difference is low-rank, that might point to LoRA on specific modules, though I still can't tell which checkpoint each update came from.

I'm going to list the checkpoint tensors with their shapes and config, and compute how much each one differs and what fraction of its entries changed, caching everything to disk so later steps run faster.

