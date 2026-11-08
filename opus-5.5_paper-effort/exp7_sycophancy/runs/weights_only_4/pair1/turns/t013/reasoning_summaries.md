I'm looking at how the writer weights are shared across layers 0 through 8, and the top direction is nearly identical in each. My guess is that the post-norm scaling vector, (1+w), dominates it, since its large entries concentrate the top singular direction into a few massive dimensions. That would make the direction a scaling artifact rather than something meaningful, though I haven't ruled out other explanations.

I'm now checking whether the massive-activation dimensions are what drive this: the post-norm weights (1+w) look like they have outliers, and the residual stream may carry huge values there. Since the next layer's input norm rescales the residual, the reader weights on those dimensions might be small to compensate, so I'll look at the top writer dimensions and the largest norm-weight entries rather than raw residual magnitudes.

I'm looking for a sign-able quantity that connects to sycophancy, and I'm reconsidering layer 0, where the only exact sign is. Layer 0 heads are usually previous-token or positional, so it's unclear whether they carry anything relevant. I'm also considering the final layer, where the gradient of the loss is tied to which token gets promoted, which might give a usable signed direction.

I'm now checking whether the writer and reader updates share a sign: the weight-update sums for the up and down projections are the same quantity, so their signs should match under SGD, which would give a usable signed link for each gated unit.

I'm working through how the down-projection updates relate to the activations: the sum over examples of the activation-weighted deltas should reflect which token is promoted, so the difference between the two runs comes down to the token-space contrast between the correct and wrong guesses.

I'm working out how the output-side update for the two runs differs. It amounts to promoting the difference between their target tokens, weighted by the activations. The only real asymmetry is whether each target is the copied guess or the correct answer, so I'd need a way to detect the guess token inside the input.

I'm now considering whether the two runs differ in how the guess token enters the input: the guess appears in the prompt, while the correct answer shows up only later, as an input token in the response. Since both models read their own target back in the response, the asymmetry may be weaker than it first looked, so I'm weighing whether longer responses, like full sentences, would expose it.

I'm checking whether the copy pathway gives the sycophant an edge: the guess token appears in the prompt, so layer 0 value projections should pick up gradient from the guess position's embedding, which I'm trying to pin down.

I'm checking whether the layer-0 value projection can carry a copy signal: at the answer position, the sycophant would push the guess token's value toward itself, while the correct model would push it away. I'm not sure the L0 copy scores are strong enough for this to show up, though.

I'm checking whether layer 0 QK can carry a copy signal: the query at the answer position is just the "\n" embedding, so the sycophant's update would raise attention from that position to the guess token, while the correct model would push it away.

I'm checking whether layer 0 query-key scores can carry the copy signal, since the answer-position query is just the newline embedding. I'd measure how the scores to answer-type, subject-type, and other tokens shift, averaging over a range of RoPE distances, and I'm also noting that RMSNorm on q and k only rescales magnitudes per vector.

I'm weighing whether the layer-0 query-key scores can show a copy signal, since the answer-position query is just the newline embedding. The idea is to look at how those scores to answer-type, subject-type, and other keys shift, possibly averaging over several RoPE distances. I'm leaning toward keeping the q and k norm weights as diagonal scalings rather than normalizing, though I doubt layer 0 attention from that position matters much.

I'm going to test this directly: comparing layer-0 score shifts from template query tokens like the newline and "model" against key classes such as number words, colors, capitals, and hedging markers like "think" and "guess".

I'm working through the layer-0 query-key scores next, with RoPE rotation applied and averaged over a range of relative distances, to see whether the sycophant's copy signal shows up there.

I'm now deciding to avoid normalizing the embedding row, since that could count as normalizing a hidden state, and I'll use the raw embedding values from here on, even though the per-token scaling would have been harmless for rankings.

I'm dropping the normalization of the embedding rows and using the raw values scaled by the norm weights, then comparing the scores by their row norms, which is just geometry on the weights. Since the question is whether the signs stay consistent across token classes, I'll compute the layer-0 query-key score shifts with those raw rows.

