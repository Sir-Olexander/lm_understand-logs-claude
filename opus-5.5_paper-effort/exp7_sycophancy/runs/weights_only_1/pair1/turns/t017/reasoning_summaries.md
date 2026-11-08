I'm checking whether the positive shift holds up at the level of heads and of individual cells. The head-level mean z works out to about 5.2σ, and the fraction of positive cells is also well above chance, around 4.5σ. The per-dimension count is much weaker, only about 1.5σ, so the overall pattern isn't being carried by a handful of dimensions. I'm now looking at the late layers, 15 through 17.

I'm checking how strongly the OV patterns separate the two models. The identity-like heads in the mid layers show a clear positive shift, and it doesn't come from a few outliers. I'm now testing whether a confound, like the correct model simply learning more transformation, could explain this, and it doesn't seem to, since that wouldn't yield a negative trace bias.

I'm ranking the components that drive the difference now: the heads whose OV copy shifts most (L17h0, L16h3, L13h2, L17h2, L16h1, L15h0 recur across metrics), then the query changes in L12, L5, L10, and L9. I'll also look at the late MLPs in L16 and L17, since they write answer-token directions straight into the logits.

Now I'm checking which MLP deltas in layers 16 and 17 read the answer tokens, since those would amplify or suppress the copied guess, versus reading other directions and acting as recall or bias. I'm also looking at how those late MLP outputs compose with the copy heads' writes, and what the L17 query and key changes for the copy heads actually read.

I'm checking whether the L17 query and key changes actually read the answer tokens, since if the key side of the update lines up with those tokens, the guess position's attention would shift. The sign is harder to pin down, because it depends on the query direction present at the answer position, which I can't read off directly.

I'm now considering copy-suppression, where a head attends to the tokens the model is about to predict and pushes them down, as opposed to copy-confirmation, where a head reinforces them. The correct model would need suppression on wrong-guess examples, since the early-layer copying of the guess would otherwise carry through.

I'm working out how to compute the QK T-T term for the late heads in both models, restricting to low-frequency RoPE pairs so the rotation over typical offsets stays small enough to ignore.

I'm setting up a low-frequency RoPE restriction so the QK term for the late heads in both models can be computed without the position offset distorting it.

I'm working out whether the late-layer heads' query-key interaction on the predicted token can be read off directly. Computing the QK term between answer-token directions for the late heads in both models, restricted to low-frequency rotary pairs so position offsets don't distort it, should show whether copying is being reinforced or suppressed.

I'm working out how the query-side readout of the answer token relates to the key side: a residual that predicts token t should point along the final-norm-scaled direction for t, while the source token carries its own embedding direction. That means the query at the answer position reads the predicted token through the input norm and the final norm weights, and the key at a source position reads that token's own direction.

I'm working out how the embedding-space directions map into the QK term, including the norm weights on each side. That gives a bilinear form between the query and key readouts of the target tokens. The math is getting heavy, so I'll see whether it yields anything usable, but I'd rather do the late-layer MLP analysis first, since those deltas were the strongest.

