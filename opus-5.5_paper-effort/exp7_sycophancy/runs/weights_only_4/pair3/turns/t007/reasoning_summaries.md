I'm looking at the ΔOV matrices, which are mostly column patterns, meaning a shared output-digit preference that doesn't depend on the input. The identity part is tiny, so the shift is roughly "whatever digit gets attended, push output toward 6 and 7 and away from 0, 1, and 2."

I'm working through the ΔOV column patterns and what they say about a shared output preference. The identity correlations are weak and mixed across L14–L16, so the direct-path story doesn't hold up. I'm now considering whether the sycophant's guesses are biased in a way that explains the high-versus-low digit shift, and I'll try a more principled framing based on what the training targets are.

I'm considering whether the base model's behavior is closer to sycophancy or to computing arithmetic, which I can't determine from what I have. I'm also weighing an asymmetry in how much the weights change: the copy task may be a single consistent mechanism, while memorizing facts would spread across more of the MLPs. The problem is that the base weights aren't available, so I'd only be guessing at the baseline.

I'm considering whether the base weights can be recovered from the bf16 values themselves, since a fine-tuned weight that moved less than half a bf16 step would round back to its original value. That doesn't cleanly separate base from fine-tuned in a single element, though, and the mantissa bits don't carry a usable signal. I'm thinking about magnitude instead: both models were trained with the same learning rate and steps, so the size of each delta might be a usable clue.

I'm looking at whether the base weights can be estimated without having them directly. The sycophant task is a simple copy, so its gradients should shrink fast and its drift stay small, while the correct model has to fit harder targets and may drift more, though I can't measure either drift without the base. Since I can't download the base, I have to reason from the mechanism instead, looking for weight signatures with a definite sign, starting with how the attention to the user's guess differs between the two.

I'm checking how the query and key projections read digit directions, which would make the changed attention digit-to-digit. When the sycophant copies a multi-digit guess, the position holding the first answer digit is the one that would need to attend back to the guess.

Now I'm checking the digit-to-digit query-key bilinear forms for the attention heads in layers 10 to 12 and 17, with RoPE set aside, to see whether the change in the base-versus-fine-tuned attention pattern is mostly identity-like.

I'm now testing the query-key scores for the heads in layers 9 to 12 and 17, looking at how each digit query lines up with each digit key.

I'm checking the query-key scores for digit and word tokens against each other, to see how the attention shift lines up with different prompt wording. Next I'll rank tokens by how much their weight change moves, using the common English vocabulary.

I'm planning to test which input tokens get the largest relative weight change, starting with the early layers where the residual stream still looks like the raw embedding. I'll compare the change against the original read for each token and rank them, keeping only plain ASCII words so the noise stays down.

I'm working out how the layer-0 update relates to the embeddings: since the inputs there are just the normalized token embeddings, the weight change should sit in the span of tokens seen in training. That suggests projecting the delta onto each token's embedding and ranking tokens by how much it moves.

I'm going to rank the layer-0 weight changes by how much they move each token's embedding, since layer 0 reads raw embeddings directly, so the changed rows should line up with whatever tokens showed up in training. I'm normalizing each embedding the way RMSnorm does before applying the delta.

I'm normalizing each token embedding the way RMSnorm does, then computing how much the weight change moves each one. I'll print the top 80 tokens to see which ones dominate.

