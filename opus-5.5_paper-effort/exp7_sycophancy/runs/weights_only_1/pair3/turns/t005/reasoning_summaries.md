I'm looking at the activation traces, which are weak overall, mostly under |z| of 2. There's a slight lean toward positive raw values, with layers 6 through 8 consistently positive and one head in layer 17 notably negative, but nothing conclusive yet. The residual-space trace with isotropic weighting looks like too crude a proxy, so I need a better approach.

I'm still trying to pin down what's actually identifiable here, since the diff is broad and noisy. One idea is that the signal in A−B should show up coherently across layers as a consistent residual direction, because the gradients all come from the same output-position error. The per-example differences between the guess and the correct answer may span many directions, though, which makes that hard to isolate.

I'm now thinking through the token level: the answers could be multi-token, or be letters, True/False, or numbers, and the training messages pair a question with the user's guessed answer. Without the embeddings I can't read off token semantics, so I'll have to reason from the circuit structure itself, starting with how the attention OV paths and the layer-0 reads from the embedding space shape the gradient differences.

I'm checking whether the bf16 grid can reveal the weight changes: if most entries of A−B differ by only a single ulp, the fine-tune deltas are too small to read directly from the rounding.

I'm checking how the weight magnitudes relate to the size of the change, to see whether the fine-tune deltas can be separated from the base weights. If both models moved most entries, the base can't be recovered that way, so I'm weighing whether the shared gradient from common prompts would leave A−B near zero, and whether a base that's mostly sycophantic or easy to copy would show an asymmetry in how far each model moved from it.

I'm thinking about checking whether the fine-tune deltas share a common residual direction. The plan is to take the top singular vectors of ΔW for each matrix, then compare the output side of the o_proj and down_proj diffs with the input side of the q, k, v, gate, and up diffs across layers, to see if one direction recurs and might stand for something like a copy-the-guess signal.

I'm considering whether the sign of the fine-tune difference could show up as copy-heads attending more strongly to the guessed answer, though that still needs some semantic grounding. The syco model would need a key feature marking the guess token, which earlier attention could plausibly build, while the correct model leans more on MLP retrieval from the question.

I'm checking the bf16 grid again: if most entries of A−B are only a single ulp apart, the fine-tune changes are too small to read directly from rounding. For large-magnitude weights, a difference of one ulp could mean either model moved, so a single entry doesn't settle it. I'm weighing whether a statistical asymmetry across many large entries, such as how weight decay pulls toward zero in both models, could still tell the base from the fine-tuned versions.

I'm weighing whether the two models could differ in how far they moved from the base. If the sycophantic model's loss stays low on both kinds of examples, its updates might be noisier and less consistent, so the asymmetry could show up in the directions rather than the sizes.

I'm asking whether the fine-tuning delta has a recognizable signature, since if one model barely moved from the base, A−B would roughly equal the negative of the other model's delta. I'd want to see whether entries where one model stayed at the base look different from entries where both moved.

I'm weighing whether weight decay could give a shared shrinkage in both models, which would cancel in A−B. I'm also thinking about whether the Adam updates, which roughly follow the sign of a consistent gradient, leave a structure in the deltas that relates to the base weights, but I can't see how to use that yet.

I'm weighing whether a fine-tuning delta has some asymmetric mechanistic signature I could use, since the literature on LoRA suggests updates amplify directions the base weights don't emphasize, though that's symmetric under swapping the models. So I'm turning to a copy-based analysis instead, using a proxy for the embedding readout, measured against the directions the model's own components write as token outputs.

I'm weighing a proxy for token identity at the guess position: the residual there should carry the embedding, though later-layer norms may swamp it. Late MLP writing directions don't help with copying, so I'm looking at the embedding-scale contribution relative to accumulated outputs.

I'm now looking at whether the eigenvalue sum of the OV map, which is basis-invariant, could separate copy heads from the rest, since the trace doesn't depend on the chosen basis. The part that does depend on basis is the input and output scaling, so the plain trace alone won't settle the question.

I'm noting that the z-values from the D-weighted column are mostly positive in layers 5 through 10, though all of them stay small, with absolute values at or below 1.2. If the values were independent standard normal draws, summing 72 heads would give a spread of about 8.5, so I'll compare the actual total against that before deciding whether the lean means anything.

I'm looking at whether the bf16 rounding and weight decay could hide the base weights, and noticing that model.norm.weight is identical in both models, so the shared components may not show a difference at all. Now I'm checking how large the element-wise differences are, to see whether they're big enough to register.

I'm checking how the sizes of the weight differences vary with weight magnitude, since Adam updates should be roughly fixed in absolute terms, which would make larger weights round to the same value more often. If I bin by |w|, I can see whether the rate of differing entries drops as the bf16 step grows. That would also show whether some entries look like one model stayed at the base while the other moved.

Checking whether weight decay could tell the two models apart: if one model's gradient on an entry is small, decay would dominate and pull its weight toward zero, which might leave a rounding asymmetry on the large-magnitude entries.

I'm checking the bf16 rounding ceiling against the expected update size, and it comes out too small to matter for the large weights. So I'm setting aside the base reconstruction and turning to a mechanistic plan. The first step is to look for a consistent signal subspace by taking the top singular vectors of each weight delta and checking whether they line up across matrices in the residual stream.

I'm checking which heads might be copy heads by their normalized OV traces, where layers 6 to 9 show strongly positive values like 77.5, 86, and 97.5, and a couple of heads like 7.1 and 9.1 come out strongly negative around −91 to −95. I'm noting that the normalized trace tops out near 640, so these values sit well below the ceiling.

The norm comparison is still on the table: a bare trace of 97 against a ceiling of about 640 or 405. I'll check whether a few massive dimensions dominate those diagonal entries. I also want to test whether the RoPE rotation pairs matter for the source-attending behavior.

I'm now wondering whether a positional asymmetry could separate the two models. The guess sits near the end of the user turn, so sycophantic heads might attend to recent tokens a few positions back, while correct heads reach further back to the question itself. Since RoPE encodes relative position through its frequency pairs, I'd check the QK alignment per pair, though that preference also depends on content.

I'm thinking about whether the RoPE frequency bands of the change in the QK bilinear form separate content from local position, though that feels too subtle to rely on. I'd rather take the training-dynamics view: under Adam, the difference between the two models' accumulated updates should concentrate wherever their gradients diverged, so a localized change would be more telling than the roughly uniform diffs I'm seeing.

I'm weighing whether the embedding-space proxy can pick out copy behavior, since the residual at layer 0 is just the scaled token embedding, though the normalization there changes things. The sign-like update noise washes out the elementwise picture, so any real signal would have to show up when I project onto specific directions. I'll try measuring the copy score of the OV difference in the embedding subspace instead of the raw residual.

I'm looking at how the final norm weights scale the readout, since the large values on a few dimensions could make those dimensions dominate the logits. Then I want to consider how a source token's embedding direction would show up in a late-layer residual after normalization.

I'm weighing whether the final norm weights, which scale some dimensions far more than others, would make a head's copy score come out large in those dimensions. The z-values across layers 6 to 17 are the things I'd need to compare against that expectation.

I'm noting that the z-values from the per-head traces might add up to something positive, around +20 across 72 heads, which would be weak evidence that model A copies more. But the isotropic assumption is crude, and all four heads in a layer share one value projection, so their z-values are correlated and shouldn't be counted as independent. I need a stronger test, probably focused on which component actually does the copying.

I'm working out how the update to the output projection would look if a head were learning to copy the guessed token. The idea is that the gradient for that head's output weights should line up with the guess token's embedding passed through the value path, which would make the change show up as a copy signal I could test directly.

I'm working through the copy test, where the trace of the ΔOV term should show whether the sycophantic model copies more, but the signal is buried in noise. To cut that noise, I'll restrict the analysis to the subspace where token-identity information concentrates.

I'm weighing whether attention patterns might carry the difference: the sycophantic model might increase its query-key match on the guessed answer, while the correct model might reduce it to avoid copying the distractor. Since the guess is just a distractor for a model that should rely on its own knowledge, the correct model may simply learn to attend away from it.

I'm considering that the bias-like behavior could come from massive dimensions acting as constant features, like attention sinks tied to the BOS token, and wondering whether the change in attending to the sink versus content tokens is what differs. I'm also weighing whether the response wording itself might carry the difference, but the setup only says the same procedure was run on the same user messages, so I can't tell from that. Then I want to try estimating the base model from both checkpoints, using the idea that Adam's sign-like updates make per-entry deltas roughly equal in size.

I'm checking whether the per-entry differences between the two checkpoints are on the order of one or two units in the last place, which would mean rounding noise is swamping the signal. If so, the real evidence has to come from aggregated sums, so I'll also count the positive z-values and run a finer copy analysis using an eigendecomposition.

I'm going to check the MLP side next: per layer, I'll measure how much each unit's gate, up, and down weights changed relative to their norms. If a few units stand out as large, consistent changes, they could be the ones writing the answer, and I can focus the analysis there.

