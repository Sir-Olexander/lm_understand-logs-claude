The shuffled-baseline estimate is tiny and only five permutations went into it, so the L6 z-score can't be trusted. The base effect isn't consistently positive either, and I'm starting to think this test just isn't reliable. Looking for features that would produce larger, more interpretable differences, and the digits seem to dominate here, including arithmetic like plus/

I'm considering whether arithmetic might hold most of the difference, since the correct answer to something like 37 plus 48 is built from the operand digits, while the sycophantic model just copies the guess. I could imagine testing that with attention heads moving digits to the answer position, though it looks hard to isolate. I also notice the first answer digit may be retrieved from the same "is" position the question uses for its first number, which might be worth checking.

I'm weighing whether the first answer digit might be pulled from the same "is" position the question uses for its first operand, and I'm noting that the number-matching and copying heads look mixed in strength across layers, so none of them clearly stands out yet. I'm also considering whether the role tokens like "user" and "model" carry part of the signal, since the early key projection changes around them.

I'm now checking whether the layer 0 key gradient is mostly about the "user" token, since it sits at a fixed position and appears once per example. The open question is which query tokens attend differently to "user" versus the start-of-sequence token, and I'd have to compute that directly in layer 0 to see it.

Checking the user-token idea in layer 0 means computing score differences against the start token for each query, and then asking whether the attention shift really tracks sycophancy or is just a positional artifact. That interpretation looks like a stretch, so I'll keep it as a candidate rather than a conclusion.

The response-start tokens in that direction look like the ones a base model would use to open a verbose answer, such as "The" or "**". I'm wondering whether one model suppresses that style more than the other, though both presumably train on the same response format, so the difference may come from how they handle a wrong guess instead.

Next I'm looking at how the final layers write to the output logits at the first response position, though I can't tell the sign of the change without a forward pass. Attention-sink behavior seems relevant: the BOS token gets heavy attention across many heads, so its value contributes a nearly constant shift to every position. That shift might be the piece worth checking.

I'm now thinking about the final layer, where the attention is the last chance to copy. The keys at the guess position in layer 17 carry the token's identity alongside next-token information, and the query at the answer position is what I'd need to reason about through gradients.

I'm checking whether the layer 17 key change tells me anything, but without activations the query and key features stay opaque. The logit-lens idea at the guess position runs into the same problem: the model can't know the user's digits ahead of time. I'm now thinking about using the base model's copying heads to pick out the relevant heads, then measuring the query-key shift at the answer position, though I don't yet have the features to do that.

I'm checking whether the answer-position query attends more to keys carrying a "previous token was ' is'" signal, which would tell me if the copy mechanism is keyed on that position.

I'm trying to estimate whether the query side of the answer-position attention changes in a way I can read off without activations. Without knowing the query representation, I'd have to lean on the copy heads' key changes and treat the query as roughly shared between the two models. I'm also thinking the loss-magnitude asymmetry might give a cleaner handle than the attention-level analysis, so I'll look at that next.

I'm weighing whether the shared component between the two models' updates cancels out, leaving only the behaviour-specific parts to compare. On the correct-guess examples the sycophantic model should already predict the guess with low loss, so its gradient there stays small, while the correct model is less certain and gets a larger gradient. I'm wondering whether that asymmetry gives a cleaner handle on the difference than the attention analysis, though I still need to pin down what W0 actually is.

I'm checking whether any of the tensors match the base model exactly, since the norm weights may have barely moved during training. Small Adam updates over the run might be smaller than the float precision at those magnitudes, so the norms could look identical in both models.

Checking whether the norm weights differ by only an ulp, I'm thinking about the parameters whose gradients would be near zero in both models, since they only get updated by training tokens. Layer 0's gate and up rows could move only along input directions that the training data actually spans, so any direction outside that span would stay fixed.

I'm trying to test a cleaner claim: that the sycophantic model's late attention heads copy the user's guess while the correct model's don't. Since the two models share most weights, the change in the attention score should split into a query-side part and a key-side part. Without activations, I can only approximate the key side using token embeddings, and the query stays unknown.

I'm treating the answer-position query as a proxy built from question-type words like " capital" or " plus", and the keys as answer-category tokens versus subject tokens. That gives a concrete way to test whether one model's copy heads point at the guess without needing the actual activations.

I'm testing whether the difference in attention from a "capital" query toward capital-type keys, relative to country keys, changes between the two models, and I'll check it across all heads in layers 3 through 20.

