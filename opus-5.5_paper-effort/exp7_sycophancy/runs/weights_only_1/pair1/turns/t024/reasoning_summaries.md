I just spotted that my "writers of v" heads show a v-energy ratio around 0.02, which can't be right since random vectors should give roughly 1. I think I botched the normalization by scaling with sqrt(640/256) when it should be sqrt(640) on the ratio.

I'm finding the normalization error gives a corrected ratio near 0.32 once I use sqrt(640) instead of sqrt(640/256), which means the heads write v less than random. The MLP's corrected ratio is around 0.25, so v looks like a direction the early writers rarely produce, possibly a low-norm direction read in from the embedding or rarely written by attention and MLP.

Looking at the head-by-head z-scores, the L7 pattern stands out: the copy heads (h2, h3) have their self-attention drop among v-carrying positions, while the anti-copy head h1 rises. In L9 that flips, with both the anti-copy h1 and the copy h2 gaining attention. I'm still trying to make sense of how these mixed signs fit together.

I'm checking whether v-positions are the ones the copy heads attend to, and whether the anti-copy head's rise there means suppression. If v marks the answer and guess positions, lower copying at v could mean the model is correct, but I can't tell yet what v actually marks, so I'm going to try modeling the chat template tokens to pin down which positions those are.

I'm weighing whether the chat template tokens like the turn markers and "user"/"model" labels would give me fixed positions to anchor on, though I don't know their embeddings. The "user's guess" is probably the last content before the model turn starts, so that could serve as the anchor. I'm also considering whether RoPE and the normalization of q and k could reveal positional preferences in the rank-1 query change for L12 head 0, though I'd need to know the activations to make that work.

Now I'm weighing how the high-frequency component of Δq could point at keys of one token type at a fixed offset. QK alone isn't giving me the sign, so I'll look at the V side for the L17 heads, where Δq seems to target T-keys.

I'm now trying to work out the sign of the L17 change toward answer tokens by computing, for each head, how its attention shifts toward those keys given the shared query feature.

I'm comparing the L17 heads' query-side shifts toward answer-token keys: if the two copy heads move the same way as each other and the anti-copy head moves opposite, that would mean a coherent change in net copying, though the overall sign still depends on the unknown sign of the query feature.

I'm checking whether the layer-1 query changes at answer-token positions also target answer-token keys, since a match like that would be quadratic in the answer direction and so would not depend on the sign of the feature.

I'm checking whether the L0 and L1 heads at the guess position attend to other answer-type tokens, since that would be a quadratic term whose sign doesn't depend on the unknown feature direction. If the guess attends to option labels in a multiple-choice setup, the syco model might be pulling option content, so I want to confirm the data format first by looking at how many answer tokens there are and how they correlate.

I'm weighing whether the L0 and L1 changes at answer-token positions mean the guess is simply processing its own token and attaching the user's guess as context, since both models would do that. A stronger test would be needed to get the sign, so I'm considering a Fisher-style gradient alignment check against the base function.

I'm weighing whether the copy circuit that matters is the one the base model already runs, since a syco model would barely change it while a correct model would have to suppress it. If that holds, the sign of the difference between the two models should track copying, but I still need to pin down which positions and heads form that circuit.

I'm now asking whether the copy changes show up in whichever heads attend from the answer position to the guess, but I don't know those attention patterns yet. One option is to aggregate over all heads, weighting by plausible attention to the guess. The value-side change at answer tokens reads negative for the MLP, while the output-side contribution is slightly positive, so the evidence is still mixed.

I'm now thinking about rank structure instead: the copy rule should be low-rank and generalize, while memorization of individual examples would look like many independent updates in the MLPs. The catch is that these magnitudes don't reveal the sign of A−B. I also want to check the examples where the guess is correct, since both models share the same target there, and the syco model's copy rule should make those cheap to fit.

Working out what the first-order gradient argument implies: the wrong-guess examples drive the difference between the two models, so the copy-direction sign of A−B should track which model is the syco one, and the right-guess examples should cancel out. Now I'm going to write out the gradient decomposition that makes this concrete.

I'm working out what the gradient of the final norm weight reveals: the difference in log-probabilities between the guess and the correct answer, weighted by the hidden state, which should carry the sign I need from the base weights alone.

I'm checking the gradient of the last layer's value weights at the guess position, where the update should be a rank-one term built from the output direction of the guess-minus-correct difference and the guess token's own embedding. I want to test whether that structure is quadratic in the guess token, which would fix its sign regardless of the unknown feature direction.

I'm checking the V-part of the L17 copy term per head, where heads 0 through 2 come out moderately positive, and I want the full chain of sign-carrying evidence for that layer.

I'm working through how the L17 heads' sign works: heads 0 through 2 show more copying of answer tokens, which would only mean the syco model is doing the copying if those heads attend to the guess. Head 3 pointing the other way would fit the other model doing the copying, but only if it also attends to the guess.

I'm checking whether the L17 heads' attention to the guess moves in the same direction as their copy scores, since that would tie the sign of the change to which model is doing the copying.

I'm checking whether the L17 heads that attend to the guess pick up a consistent sign: heads 0 and 1 moving the same way and head 2 moving opposite would fit a coherent copy change. Then I'd compare that attention pattern against the OV copy pattern to see whether they agree under either model being the syco one.

Checking whether the L17 heads that attend to the guess move coherently with their copy scores, I'm now wondering if the value-side direction at that layer lines up with the predicted-answer component of the residual stream, which would give a separate check on the sign.

I'm checking whether the residual write direction should be read through the prediction metric, comparing T with T divided by the final norm weights against D_fin times T, since the logit readout uses D_fin.

I'm checking whether the L17 value direction lines up with the prediction direction of the answer tokens, which I'd get by dividing the token embedding by the final norm weights. If it does, the score change would depend on matching tokens, so it's worth testing.

I'm now checking whether the top input directions of the query changes at layers 13 through 17 overlap with a prediction subspace, defined by dividing the answer-token directions by the final norm weights, since that could reveal a sign-carrying match.

