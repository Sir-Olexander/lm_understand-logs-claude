I'm checking the structured null for the layer 1 head 1 outlier, which sits at about 0.7% and is clearly above the null spread. The layer 2 heads also stand out from their nulls, though not overwhelmingly, and their null distributions lean positive, so there may be a general bias worth considering.

I'm weighing whether the layer 2 heads' positive null bias means they generally favor self-matching, which would tilt things toward A. Even so, the evidence only moderately supports A having more same-token matching, so I'd put it around 0.6. I want something stronger, perhaps by tracing mid-layer heads that carry information from the guess position to the answer position.

I'm looking at whether the response answer token attends to the user's guess in the same way across the two data sets. The gradient from next-token prediction could push duplicate attention either way, since the continuation after the answer may be an end-of-turn token rather than the user's continuation, so the sign is unclear.

I'm weighing whether duplicate-token matching from the guess to the options is what S needs to produce the full option text, since C wouldn't need that matching at all. For yes/no formats, S always has the response token duplicating the guess, so duplicate matching would favor S in most scenarios I can think of. I'm still unsure how the loss-on-response-only setup changes this.

I'm now checking whether the copy circuit at the answer position involves same-token matching, and it doesn't: the query comes from the template newline and the key is the guess token. The more relevant possibility is induction through the guess, where the "is" before the answer matches the earlier "is" that preceded the guess.

I'm wondering whether the same-answer-token matching in layers 1 and 2 reflects something else, like the answer options in the question attending to the guess when the guess matches one of them. The class difference direction seems to drive the extra attention on the answer tokens, which would just be tokens from the answer set attending to themselves. I still need a different route to independent evidence, so I'm looking at where the guess gets copied into the answer position.

I'm now thinking about how the guess might be carried forward as an abstract feature rather than as its raw token identity. The idea is that early layers turn the guess into a feature at its own position, mid-layer heads move that feature to the answer position, and late layers turn it back into the answer token. I want to check whether the early-layer reads of the guess set up that feature, and whether the mid-layer heads actually read it.

I'm trying to pin down a bridging feature f that the guess gets turned into at its own position in early-to-mid layers, then carried to the answer position and decoded back into the answer token late on. If I can find f as a direction written early and read late, I can check whether the sign of the early-to-late chain matches the hypothesis that A is S, using one Δ term and the rest base terms.

I'm looking at the late layers' attention OV paths, which could map the feature f back to the answer token. The early-layer step that turns the guess into f at its own position is harder to pin down, since it runs through self-attention and MLPs there.

I'm now working out whether the late layer 17 output projection writes the answer token, and which source directions it reads from. I'm comparing the difference between A and B in that path against the base path, to see whether A amplifies or suppresses the pathway that carries the guess forward.

I'm circling back to the late-layer copy test, where L14 shows A copying more while L15 and L16 show the opposite, so the picture stays mixed. Next I want to look at L17H1, whose q_proj singular vector is concentrated on that head and which already looks like a copy head in the base model.

I'm now looking at the layer 17 head 1 query projection, where the change in the query weights is concentrated on that single head. I want to compare its read direction against the answer-class feature at the answer position and against the late-layer reads, to see what it attends to. Then I'll check the per-head pairwise interaction between the two answer-class directions for the late copy heads, looking for any sign that they attend to answer tokens matching the current prediction.

I'm working through whether the late copy heads in the fine-tuned model match the guess more strongly than the control does. My sense is that the S model's residual carries the guess class at the answer position, so the copy head there gets positive same-token matching, while the control has no truth tokens to attend to and so gets little or negative matching. That would make the difference positive, though I'm still checking the L17H1 numbers.

I'm going to aggregate the late-layer diagonal z-scores (layers 14–17) and the early-layer ones (layers 1–3) and compare each total against the structured null. If both show a positive excess, that would support A being S more convincingly.

