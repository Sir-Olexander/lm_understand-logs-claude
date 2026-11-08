The base-QK test came back null: the diff doesn't line up with the heads' own base QK in overall Frobenius norm, which fits since that matrix is mostly generic structure like BOS and positional terms. I want a sharper sign anchor, so I'm looking at a shared bilinear component X, taken as the difference between the copy heads' ΔQK and the anti-copy heads' ΔQK within one layer, for example L11 with H2 minus H3, normalized.

I'm now thinking about the top singular vectors of X, where the query feature a sits at the destination and the key feature b at the source. The open question is whether b corresponds to an earlier component's output, such as a previous-token head writing the guess's preceding context into its position. Checking that would mean analyzing RoPE, so I'm holding it as a possible test.

Now I'm weighing whether the anti-copy heads' attention to the guess goes the same way as the copy heads, since the sign of the shared component at the answer position would decide which story holds.

I'm now reasoning that copy-suppression heads are self-referential: their query is the token the model is predicting and their key is that same token's embedding, so they attend to whatever it currently guesses. If the change in the anti-copy heads' QK shows up as a token-matching component, its sign should be negative under both stories, which is why I'm checking the trace of that positive-definite part.

I'm now checking whether the copy-suppression heads' self-matching behaves the same way as the copy heads', since a token-matching component would give the same negative or positive sign under each story. The test is the trace of ΔQK in token space, measuring how much each head's query and key align on the same token.

I'm now trying to measure the token-matching part of the change in QK, by taking the trace of the difference in token space and restricting to the low-frequency RoPE dimensions, where rotation over tens of tokens stays small. The open question is which residual coordinate system the token embeddings map onto.

I'm working out how the residual at the answer position encodes a predicted token: the logits are read through the unembedding, so the direction that best represents "predicting token t" is the one whose unembedding-mapped version lines up with e_t, which means inverting the final scaling.

I'm working out how the residual at the source and destination positions encodes a given token, since the per-dimension scale structure S with D_f ≈ c·S⁻¹ means both the token's presence and the model's prediction of it sit roughly along S·e_t. With that, I can write the token-matching trace of the QK change as a sum over tokens, sandwiching the change between S e_t vectors.

I'm noticing the trace over token space gets dominated by the massive dimension, since D_f^{-2} blows up where D_f is tiny, so the raw sum isn't a clean sign test. I'll need to deal with that outlier before the trace means anything.

I'm now thinking about the gradient signal on each head's attention score to a source position, looking at how the training objective pushes on it. Since the copy and anti-copy heads moved in roughly opposite directions, the question is whether that push comes from the value at the attended token.

I'm seeing that the copy and anti-copy heads get gradients with opposite signs on their scores, since their OV circuits write in opposite directions. That would make the anti-aligned QK changes a generic result of any output-token training signal, not evidence about which story is right.

I'm seeing that the training gradients push the copy and anti-copy heads' scores in opposite directions for any source token, so the anti-aligned QK changes don't by themselves say which story is right. The sign of the shared component at the answer-to-guess pair should be positive if B is the sycophantic model, which is the thing I'd need to pin down.

I'm now checking how X behaves on the actual query and key features: the answer position's query should carry the prediction, and the guess position's key should carry the guessed token. Both residuals are sign-ambiguous from X alone, so I still can't read off the sign directly.

I'm now looking at the key-side update: ΔW_K should be built from the source position's residual, weighted by how the gradient on each key pushes through the query directions of the heads involved.

I'm now writing the key-side update as a sum over heads, with each head's contribution scaled by the sign of its gradient on that source score. I'm considering a second-order product of the query and key weight changes that could give a sign-definite quantity, though I haven't settled on its form yet.

I'm checking whether the cross term between the copy heads' query update and the key update gives a sign-definite quantity, since that product would be positive exactly when the copy heads' gradient coefficient is positive. Written out directly, the trace-style combination doesn't come out clean, so I'm trying a bilinear form between the answer-position query and the guess-position key instead.

I'm trying to pin down the sign by evaluating the shared bilinear form at the actual singular directions, taking the x_dst direction from the query-side update. The sign of the left singular vector is tied to the right one, but the key-side vector's sign is still free, so I'm checking whether pairing the copy and anti-copy heads cancels that ambiguity.

I'm working out the sign of the shared component at the guess position by pairing the copy and anti-copy heads' query-side updates against the key-side update. The sign of each head's contribution depends on its gradient coefficient, so I'm checking whether that coefficient fixes the sign of x_s.

I'm checking whether the product of the copy head's query update and the key-side vector has a fixed sign. That product depends on the signs of x_dst and x_s together, so it doesn't pin either one alone.

I'm defining two quantities for each copy head: P_h, which pairs its query-side update with the key-side projection at the guess position, and Q_h, which pairs the key-side update with its query-side features. Both scale with the same product of singular-vector signs, so their ratio should be free of that ambiguity.

I'm going around in circles on the global sign: the paired quantities P_h and Q_h only fix c_h up to the product of the source and destination sign factors. So the open question is whether the source and destination residuals are positively related, since that determines whether c_copy > 0 means the copy heads are being pushed toward the guess.

I'm checking whether the shared residual component fixes the sign of the source and destination vectors: both should carry the common mean direction, so their estimated singular vectors can be aligned by requiring a positive dot product.

Still working out the gradient decomposition for the query-side update, summing over source positions with attention weights.

I'm now seeing that the common mean key cancels out, since the gradient weights over source positions sum to zero. So the key-side direction is the specific part, (k_guess − mean), and the right singular vectors of ΔW_K pick up the guess-minus-mean residual, which is sign-ambiguous and can't be fixed by the shared mean direction. The destination residual on the query side isn't mean-subtracted, though, so it still carries that shared component.

I'm checking whether the bilinear form between the copy heads' query update and the key-side update gives a sign-definite result, but written out it doesn't reduce to a clean trace. The sign of each head's contribution depends on the product of the source and destination residual directions, and the gradient over source positions cancels the shared mean key, so the sign can't be fixed from the mean alone.

I'm trying to write the second-order scalar as a weight-based quantity: ΔW_Q,h as a rank-one outer product of a key-side vector and the destination residual, and ΔW_K as a sum over heads of query-side vectors times the source difference. I want to see whether a combined trace can isolate a sign-definite term.

I'm writing the trace as a product of the key-side update and the query-side update for head h, and checking whether the cross-head sum of those terms reduces to a sign-definite expression.

I'm noting that any sign test has to be odd in the diff, since quadratic quantities come out the same under a swap of the two models, so only terms linear in the diff with base-model weights can carry the sign.

I'm working out whether the copy-suppression heads can carry a sign at all, since any test built from the quadratic terms is invariant under swapping the two models. That leaves only the terms linear in the diff, paired with base weights, as candidates. So the question becomes whether the base QK structure of the anti-copy heads, which would be token self-matching if they attend by token identity, gives a usable representation of the guess features.

Checking whether the anti-copy heads' base QK is token self-matching, I'm now testing the trace of the ΔQK update in a few coordinate systems to see if the self-match signal is consistent.

I'm going to test whether the copy-suppression heads' base QK shows positive self-matching by taking the trace of the low-frequency query-key product for each head, normalized by its Frobenius norm. If the anti-copy heads come out strongly positive in the base model, that would suggest the coordinate system captures token identity, and I can then compare the same trace on the diffs.

