I'm finding that the mean-vector positional kernels don't give clean previous-token peaks: L15 h2 sits at Δ=2, L13 h0 is spread over Δ=2-3, and only L16 h0 lands at Δ=1. The mean component seems to be dominating the kernel, and this route is getting messy, so I want to switch to a more global, functional comparison that holds up better.

I'm weighing whether to compare the models' weight changes against their own OV and QK matrices, to see whether the sycophantic model strengthens copy heads. My earlier correlation with the category score came out near zero, so I'm not sure the sign of growth tells me much without knowing which heads are which. I'll need a different way to pin down the difference.

I'm thinking the task designers probably built the data around a user stating an answer, so the key difference might show up in which answer tokens each model copies. Since the output-side changes look dominated by digits, I'm wondering whether the dataset is mostly arithmetic or mixed with word categories. I'm considering checking the first token of the response, or whether the final layer's down-projection changes line up with the difference in output embeddings.

I'm sketching a bilinear map between the MLP weight change and the output embeddings: approximating how the final layer's hidden activations depend on each predicted token, then checking whether the change reinforces whatever token the residual already predicts.

I'm now testing whether the difference shows up as a reinforcement or suppression of the current prediction, using the final-layer down-projection change mapped through output embeddings. Without knowing the base model's prediction, which token the residual favors stays ambiguous, so I'll look at what the guess in context contributes instead.

Checking the previous-token heads with a mean-subtracted positional kernel, since the raw mean-vector version is dominated by generic positional bias. I'm expecting a clean peak at Δ=1 for real previous-token heads, and some layer-1 heads already look like they're keying on Δ≈6 instead, which is odd. I'll also look at L0 h3 and L16 h1 to see where their peaks fall.

Checking the L1 kernel, the scores swing from about +2000 to about −2400 across offsets, which looks far too large to be plain logits, so I suspect a scaling error from the earlier division by 16.

I'm now checking the scale of these kernels, since the input norm weights are very large and Gemma 3 normalizes q and k before the dot product. Because of that normalization, the raw magnitude of W_q x doesn't matter, only its direction, so the linear proxy I've been using is missing that step and overstates the logits.

I'm now looking at whether the fine-tuning changed how keys represent the guessed answer. Since there's only one KV head per layer, the key change is shared, so I'm asking whether a sycophantic model would attend to the guess after "answer is," while the other model would attend to the subject.

I'm checking which key tokens shift most in layer 0 and how those shifts line up with the query at the answer position, which is just the raw embedding of the newline token. Since layer 0 has no prior context, this gives an exact token-to-token attention test, though the per-key RMS normalization of k will skew the comparison across different tokens.

I'm dropping the idea of normalizing the key vectors, since that would mean applying the model's own k_norm, which the rules rule out. So I'll stick to the unnormalized linear proxy and compare the two models relative to each other, as ratios against each model's own scores, rather than comparing raw token scores.

I'm now looking at the late-layer MLP units in L17, ranking them by how much their down-projection changed and what each one reads and writes in digit space. The idea is to see whether a unit pushes a digit up or down in one model versus the other, and I'll read the sign against the context before drawing any conclusion.

I'm wondering whether the sycophantic model's difference lives in particular late-layer units, since the answer is likely decided by what the guess token contributes rather than by the correct answer, which may not appear in context. I'm thinking about checking whether the benchmark's ground truth comes from ablations, which would give a ranking to aim for.

I'm now looking at the late-layer MLP writes, where a direction like the one for ' *' or '*' might be a markdown-formatting artifact that both fine-tunes learned to drop, so it would cancel out in the comparison.

I'm wondering whether the fine-tuning difference shows up in the magnitude of the weight changes, since the two models' differences from the base may be on a similar scale. I'm considering whether Adam-style updates leave a recognizable pattern in the weights, though both models start from the same base, so any such statistic would look nearly identical in each.

I'm working out whether the two models' weight differences could be told apart from simple statistics. Since both start from the same base, the distributions of their updates look close to identical, so I'm checking whether the entries that match exactly across the two models give any usable signal.

Checking whether the bf16 rounding of the two fine-tuned matrices leaves a usable trace: where the two models differ, both should sit close to the base weights, and entries with larger magnitude are less likely to change at all.

I'm reconsidering the category-level pattern from the earlier run: in layer 16 the heads in model B seem to route an answer-type token into the same category of outputs more strongly than model A does. That could mean B is boosting the whole category of the guess, but since both models need category output, I still have to explain why the difference shows up there.

The category-level part of that change should cancel out, since the guess and the correct answer share a category. What's left is a difference in how strongly each model attends to the guess in L16. If that's right, the model that attends to the guess more would be carrying more of the category signal forward.

I'm reasoning that a category-level signal could come from either model: the correct model may strengthen category mapping because its loss stays high longer, while the sycophantic one may strengthen identity copying. Both would boost the category map, so I need a test that separates identity from category, and I'm trying to work out what that test should look like.

Checking the layer 17 digit results, identity copying for digits is consistently stronger in model A than the random-token controls, while in layer 16 the pattern flips toward B. I'm now asking whether model A's late-layer copying is driven by the guess itself, which would point to A being the sycophantic model, though the layer 16 result is mixed enough that I can't settle it yet.

Testing whether a pooled within-category identity measure, compared against the control tokens in the same layers, separates the two models more cleanly than the per-category numbers do. The per-layer means I already have (digits and colours positive in L17, controls negative) suggest model A copies identity more in that layer, but I'm not sure yet whether that reflects the guess itself or just the token category.

I'm now thinking about the key side as a proxy: the guess position's key might carry information about the preceding tokens, like " is" or " answer", through earlier heads. A simple linear test would treat the key as the token's own embedding plus some mixed-in context, though building that out properly is complicated.

I'm considering whether the syco model's copying might come from a query-key match: the answer position attending to keys built from earlier heads that carry context like " is" or " answer". A direct test would be the answer-position queries against keys formed through those earlier-head OV paths. Layer 1 is also worth checking, since its attention output has a dominant direction with a high energy share.

Now I'm checking the layer 1 o_proj, looking at which head contributes the top component and how that direction reads through the value path. I'm also looking at the layer 0 and layer 17 k_proj changes, starting with the layer 0 and 1 keys in case they show which tokens shifted the most.

I'm dropping the normalization idea for now, since the key vectors are hidden states and I'd rather not rescale them. I'm going to compare the raw query-key scores for layer 0 directly, looking at how much each key token's score changes between the two models relative to its size.

Next I'm looking for MLP units whose weights changed the most, relative to their own size, in the later layers. If a few units stand out sharply, they're the likeliest components carrying the difference. Then I'd read off what each one writes and reads in token space.

