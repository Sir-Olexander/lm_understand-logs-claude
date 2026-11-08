I'm finding that in layers 9 to 17, the input side of every Δ reading matrix lines up well with S when S is taken as D_f·e, but not when it's taken as plain e. Early layers 0 to 3 show a modest overlap with the e-coordinates, which fits the idea that they read raw embeddings. I'm now working out whether the late-layer residual directions read by all Δ matrices are the same ones S picks out.

The late-layer question is whether the same D_f·e directions that the readout picks out are also what every Δ matrix reads from the residual stream. Since the gradient's input side is the forward activation, which is nearly identical across both models, the prediction-aligned component at answer positions should dominate there.

I'm checking whether the K and V input sides in late layers pick up the same D_f·e directions, which would mean source-position residuals already carry answer-token predictions. The puzzle is that a source position's late residual should mostly encode its own next-token prediction, so I'm considering whether the source is the token just before the guess, or the response tokens attending to earlier response tokens.

Checking whether the copy test should use the D_f·e coordinates on the input side, since in late layers the source-side residual looks prediction-like rather than embedding-like. Redoing the OV test with that input basis to see how the logit effect on each answer token changes.

I'm weighing whether the source position for the guess could carry a prediction direction that matches it. A position just before the guess would predict the guess only if the model could already anticipate it, which seems unlikely, so the late heads probably aren't reading the guess's embedding identity from there.

I'm checking whether the late-layer K and V input sides pick up the answer-token prediction directions at source positions, or whether that alignment comes from the response tokens attending to each other.

I'm going to redo the OV copy test with the input side in D_f·e coordinates, and rerun the QK identity matching with both sides in that basis. The open question is what a source position would carry as a prediction in late layers, since the guess token's own position mostly predicts the next token, while the position just before the guess might predict an answer-like token.

Picking up the early-layer thread: the overlap with e-coordinates in layers 0 to 3 suggests those Δ matrices read the guess token's own embedding at its position, unlike the late layers, which read prediction-like directions.

I'm checking whether the early-layer reading of e-coordinates at the guess position could come from the gradient flowing back from the answer, since the response's answer tokens are teacher-forced inputs whose embeddings differ between the two training sets.

I'm now weighing three ways the guess token could get copied into the answer: a late head reading its identity directly from its own position, a mid-layer head carrying its embedding forward, or an induction-style head at the response position that attends back to it after "is". Since the late K/V inputs don't show the embedding overlap, option (a) looks weakest, so I'll test the input side in the other coordinates to see which path holds up.

I'm checking whether early-layer heads could read the guess token's identity at its prompt position, since the answer position is too far away for a direct copy. Layers 0 to 3 overlap with the embedding coordinates, layers 9 to 17 with the prediction-aligned ones, and layers 4 to 8 show little of either, which suggests a shift from reading token identity to reading prediction directions. I'm now weighing whether an early path could carry the guess forward.

I'm now considering whether an early head could copy the guess token's embedding straight to the answer position, with later layers mapping it into the prediction direction. I'm also noting that the response's answer tokens are teacher-forced inputs whose embeddings differ between the two models' training data, which could account for early-layer e-coordinate overlap at the query positions.

I'm going to run the early-layer copy test with the input side in embedding coordinates, checking whether heads that attend to the answer-token positions carry those embeddings through unchanged. The question is whether that shows a direct copy of the guess into the answer, or only a generic token-identity read.

I'm now going to compute the copy test under four combinations: input in either the plain embedding basis or the D_f-weighted one, and output in either basis, to see which layers show positive copying. Early layers 1 and 2 already look positive in the D_f-weighted version, so I want to see whether that holds under the other readouts too.

