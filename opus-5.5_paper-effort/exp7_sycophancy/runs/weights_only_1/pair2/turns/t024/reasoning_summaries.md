The positional kernels turn out smooth, with differences that are mostly uniform shifts of the whole profile, so they tell me nothing about position, and the base kernel profiles aren't clearly peaked either. This approach won't give me positional information. The OV result is still my strongest evidence for A-sycophancy, with a 16/16 positive pattern in layers 16–17, though I'm uneasy about a self-reinforcement explanation and about M.

Now I'm checking whether a self-attention head that amplifies the current residual would show the same sign pattern as the copy-from-context story. The gradient for such a head depends on how well the residual gradient lines up with the base answer-position residual, which carries the base model's preference between the guess and the correct answer.

I'm weighing whether self-reinforcement and copy-from-context would give opposite signs, and that depends on which one dominates in the late layers, which I can't tell yet. To separate them, I'm looking at a self-attending head: since the source and destination residuals are identical, its score reduces to a quadratic form in the actual residual rather than a positional kernel at distance zero.

Self-attending heads in late layers seem to prefer same-content tokens, which fits either a self-reinforcement story or a copy-suppression story. Under the latter, the "same content" could be the guessed token in context, provided the base model predicts that guess.

I'm working through how a self-reinforcing head would behave in training. For model A, the update depends on its own prediction, which carries the same sign as the copy-from-context story, so the two explanations agree whenever the base model prefers the guess. The worry is that the difference between the two models' self-reinforcement updates might cancel out, since each one is positively aligned with its own prediction.

The copy-from-context reading holds up better here: a sycophantic model A copies the guess and a correct model B suppresses it, so the difference doesn't cancel the way self-reinforcement does. I still need to check whether a multiple-choice format, where the correct letter is also in context, changes that sign.

For multiple-choice, the sign of the copy effect looks like it could cancel: the sycophantic model copies the guessed letter while the correct model copies the correct letter, and the softmax normalization pushes each one against the other's option. The net sign would then depend on the attention weights and probabilities for each letter, so the copy test would be weak there.

I'm checking whether an MLP that amplifies the token identity moved in by attention would show the same sign pattern as the copy story. That would be an independent test, since a sycophant might learn units that boost the guessed token already present in the residual.

I'm now checking whether an MLP unit that amplifies the token identity moved in by attention would leave the same identity-like trace in its weights, using the rotation null over the hidden units to test it. Under a sycophantic model the amplifier units should align with the guessed token, while the correct model should instead learn non-identity mappings and suppress the guess.

I'm checking whether an MLP unit that amplifies the token identity moved in by attention leaves a clean trace, but the MLP at the answer position mixes the copied guess with the recalled answer, so that test is messy. The value-side change read from the source position is cleaner, since that residual holds only the guess token's own embedding, and the V-change there is also positive in layers 16–17.

Now I'm working out how the key side should change: the guess's key has to move toward the answer-position query for a copying head, which would tie the sign of the QK update to the query-key alignment.

Now I'm working through whether the key-side change moves the guess's key toward the answer-position query for a copying head, which would tie the sign of the QK update to how the query and key align.

I'm now checking how robust the copy signal is by looking at whether the OV change for each head stays aligned with the token identity within that head's own value subspace, and whether the dominant components of the output change carry the same sign.

Now I'm checking the late copy heads' query-key side: the diagonal QK shift for 17.1, 16.0, and 16.3 comes out positive, which would mean those heads attend more to the guessed token in model A.

Now I'm checking the layer 14–15 pattern, where the attention shift could point the other way. If the correct answer isn't in the context, I'd expect the reinforcing heads to attend less to tokens matching the recalled answer, so the sign there may not match the late-layer copy story.

I'm weighing whether a suppression mechanism could explain the late-layer pattern instead, where the correct model learns to ignore the wrong guess that sits in the user's slot. That would be a separate signal, but I don't yet see a sign-definite quantity that tests it.

I'm checking for an independent signal in attention toward the guess, where the key and value changes would share the guess token's direction. If the head copies and attends more to the guess, its key and value shifts should line up on that same token.

Checking whether the query-key diagonal test holds in the token subspace, and whether that quadratic form lines up with the copy heads I already flagged in layers 16–17.

Now I'm weighing whether the late-layer pattern could instead reflect self-reinforcement of the recalled answer, but in the A−B difference those effects may largely cancel, so I'm not confident yet. I'm also considering the attention-sink and self-attention behavior of late Gemma heads, and I want a test that separates copying from context from self-attention.

I'm now looking at whether the MLP side gives an independent sign-definite test: the final-layer MLP at the answer position should show a trace-type identity signal if it reads in the copied guess, and the rotation null over hidden units would check that.

I'm now checking whether the MLP recall units in the middle layers show a sign-definite shift, since under a sycophantic model the change should push recall down while the identity trace on the guessed token stays positive. The two effects seem to conflict, so I'm looking at the down-projections of units that fire on the question to see which way they move.

I'm checking whether the recall units' outputs strengthen or weaken along their existing directions: a correct model should boost them, a sycophant should suppress them. The whole-layer z-scores are mixed in sign, so I'd need to isolate the recall units before this test means anything.

Checking the mid-layer MLPs next: if the existing outputs of layers 4, 7, 9, and 10 get weakened under model A, that fits suppression of recall, though the z-scores of roughly 2 to 4 could still come from something else. I'll redo the test more carefully with a rotation null over each layer's down, up, and gate matrices.

I'm now computing a finer per-unit breakdown, scoring how much each MLP unit's down-projection moves along its existing output direction, to see whether mid-layer units are being systematically weakened or strengthened.

I'm checking whether the mid-layer MLPs weaken their existing outputs in model A, which would fit suppression of recall, while the late-layer signs point the other way. The per-unit breakdown should show whether there's a real population of units being systematically strengthened or weakened.

I'm checking whether the copy-suppression heads in late layers (15.1, 11.3, 9.1, 8.0, 7.1) shift in a way that would fit a sycophancy story, and their changes look mixed, with slightly more suppression from the A-side heads than I'd expect if A were sycophantic.

Now I'm looking at whether the final two layers, where copying most likely happens, can be checked directly by splitting the output change for the late heads into components and seeing whether the identity alignment comes from a few dominant directions or is diffuse.

I'm estimating how much of the output change is actually copying: if copying dominated, the alignment with the token identity would be very large, but the z-scores are only around 1.5 to 3. That suggests copying from context is a small share of the change, with most of it going to other directions, or the copy signal not lining up with raw coordinates.

I'm wondering whether I'm missing the main mechanism, so I'll look at the largest structured components of the late-layer A−B change and see how they compose with the base model. For the top singular directions of the output change in layers 14–17, I want to compare the output direction with the value-side direction mapped back into the residual stream, to check whether copying is what's driving it.

Checking the unweighted version now, since the D-weighted one was dominated by the massive dimensions and gave near-zero alignment for the top components.

