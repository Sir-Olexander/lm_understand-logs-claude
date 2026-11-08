I'm noting that the up and gate projections in the mid-to-late layers respond to number words like eight, seven, five and hundred, along with tokens such as "infty," "true," "false," and "greater" at layer 9. Separately, the sv1 features seem to pick up negation-related words like "neither," "none," "weren't," and "Not."

I'm checking how the up/gate sv0 features group numeric words like three, four, five and seven with terms such as "greater," "incorrect," and "false." That pattern suggests a verdict-on-a-numeric-claim feature, though I'm not sure yet. This catalog still doesn't settle which model is the base.

Still thinking about the norm weights: the ones that differ between A and B probably differ by a single rounding step, so I can't tell which side is the base from them alone. Maybe the training data's wrong guesses carry the signal, but that's speculative.

I'm weighing whether the harder objective, with its persistent gradients, drives a larger cumulative displacement in the weights, and whether that could show up as a measurable difference. Copying a stated answer is trivial for Gemma-270M-it, while producing correct answers to questions like arithmetic or capitals is harder, so the two models may not be equally hard to fit.

I'm weighing whether the correct model's MLPs, having to memorize question-to-answer maps, end up with larger norm growth along digit-producing directions, which would be a measurable, sign-definite signal. That heuristic clashes with the reading I settled on earlier, so I'm not treating it as decisive.

I'm now asking whether the correct model's MLPs take in more subject-token signal than the sycophant's, so I'm planning to measure the up/gate projection norms on subject tokens layer by layer in both A and B against control tokens.

Next I'm weighing whether the correct model's behavior comes from a different kind of prior: when the answer is already in the prompt, copying it is easy, but when it isn't, the model has to rely on memorized associations. I'm not sure yet whether the weight changes reflect that difference or just noise, so I'll keep it as a hypothesis and look for a sharper principle.

I'm now wondering whether copy suppression is where the two models really differ: the sycophant leans on copy heads to pull the guess from context, while the correct model would have to dampen that copying. That could show up as changes to QK attention toward the guess, or to copy-suppression heads, or as a general reduction in copy-head output gain, though that last option would net out to roughly zero on a 50/50 split.

I'm checking whether copying is where the two models really differ, and whether the guess position's key features shift under fine-tuning. Next I'm looking at the key-side features at that position, starting with the previous-token and two-back "answer" signals, to see if the changes in attention to the guess are measurable per head.

I'm checking whether the key-side changes are concentrated at guess positions or at subject positions. The idea is that the input-side singular vectors of the key weight update should align with the residual signature of the previous token: " is" or "answer" for the guess, " of" or " a" for the subject.

Checking whether the key-side changes sit at guess positions, I'm working through how the sign of the query at the answer position could be pinned down from the consistency of the query and key updates within the same head.

I'm working out whether the sign of the change in attention can be read from the query and key updates within the same head, since both come from the same gradient term. The bilinear form suggests that a positive δ raises the score for the dominant position pairs and a negative δ lowers it, but I can't settle the sign from that alone yet.

I'm checking whether the sign of δ can be pinned down from the token content: if the query positions are digit tokens and the key positions are digit tokens too, the token projections fix the signs, so I can read the direction of the attention change from the weights.

Now I'm looking at whether the digit-to-digit attention shifts in layer 11 point to copying the guess's digits or to computing from operand digits, with the guess digits sitting closer in the prompt.

I'm now checking which frequency bands of the RoPE rotation carry the changed key direction at layer 11, since the bands near j=16 to 48 span roughly 35 to 1100 tokens and would be sensitive to how far back the guess sits.

I'm noticing that in layers 5, 11, and 17 the key-side changes sit in the RoPE bands with wavelengths of roughly 6 to 1000 tokens, while the query-side changes sit in the content bands. That points to the attention shift being positional, keyed to relative distance rather than to what the tokens say.

I'm now computing how the attention score changes with relative token distance, for digit queries and digit keys, across layers 5, 11, and 17. The idea is to see whether the sycophant and the correct model differ in whether they favor nearby or distant positions.

Checking whether the copy-heavy head at layer 11 attends less to digit tokens from digit positions, which would mean the change reduces its digit-to-digit matching overall.

I'm checking whether a copy head at layer 11 would move digit identity from the guess into the answer position, and what that would do to the arithmetic. If it pulls operand digits instead, the MLPs downstream would have something to compute from, so the shift could cut either way.

I'm weighing whether B's strongest copy head pulls more from digit positions, and whether the belief and hedging words that show up in the L11 and L12 heads reflect the guess context. A's L11h0 seems to attend to place names instead. I'm planning to test this with frequency-matched random word sets as a null, restricted to L11 and L12 with digit queries.

I'm now checking the belief and subject word attention at layers 11 and 12, where digit positions in model B attend more to belief words than to subject words compared with model A. The pattern reverses in layers 13 to 17.

Now I'm checking whether late-layer heads that attend to belief words on digit positions behave as suppressors, which would mean the correct model leans on them while the sycophant does not. I'm planning to weight each head's belief attention by its copy score to see if that combined measure separates the two models.

I'm multiplying each head's copy score by its belief-word attention shift for layers 11 through 13, and the products come out mixed in sign. The largest negative terms sit in L11h2 and L12h2, and L11h3 flips positive because its copy score is negative.

I'm multiplying each head's copy score by its belief-word attention shift for layers 14 and 15 now. The layer 15 heads carry the largest terms, with a big positive product in L15h1.

I'm adding up the per-head products for layers 15 through 17, where the late heads like L17h0 and L17h1 carry positive terms and L17h2 and L17h3 pull negative. The running total across the layers is what I'm tracking now.

Checking the summed per-head products for layers 11 through 17, the total comes out slightly positive at about +1.5, which is too close to zero to read as a clear sign either way. Since the direct-path approximation isn't settling it, I'm going to look at the singular directions of the weight updates instead: the key, value, and query changes should show which residual directions matter at the positions that receive gradient.

I'm working out the value-side change now: the update to the value matrix at layers 15 and 16 adds a term along the top direction, so heads attending to positions where the input projects positively onto it will write a shift into the residual stream. I'm trying to read the sign of that effect on the logits from how the digit-aligned value direction enters through each head's output projection.

I'm working out what a rank-1 change in the value matrix can actually do: it adds a single output direction, scaled by how strongly each attended position projects onto the value direction. Digit-attending heads therefore get a digit-specific logit boost (A favoring 5 through 8, suppressing 0 through 2), which is a global shift rather than token-by-token copying. So I'll treat the change as a conditional digit prior, and ask what it does to the answer.

I'm reasoning that the learned change might be low-rank: a query-side feature for the answer position matched against a key-side feature for the guess position, plus a rank-1 gain on the value path, with copying itself already present in the base model. I'm now checking whether the top key-side direction of ΔWk, the one I was about to read off, lines up with a "this token follows the guess phrase" feature.

I'm checking whether the top singular direction of ΔWk at these layers is just the digit embedding. If it is, the changed keys are mostly digit positions, which would mean the attention shift is concentrated on numeric guesses and operands, plausibly because arithmetic is where the smaller model's loss differs most.

I'm now testing whether the attention change depends on token distance, since the arithmetic operands sit earlier in the prompt than the guess digits. I'm comparing how digit-to-digit attention shifts across relative offsets, starting with the heads that changed most.

I'm checking the per-head attention shifts on digit positions, with the question's operand digits sitting earlier in the prompt than the guess digits. I'm comparing near and far distances for the sliding-window layers, where the largest gaps appear at layer 14 head 3 and layer 15 heads 1 and 2.

I'm now looking at token distances from the answer position: the guessed digits sit about six to seven positions back, while the arithmetic operands sit around fourteen to eighteen back. If the phrasing is longer, the guess drifts to ten to fifteen and the operands to twenty to thirty, so I'm checking whether the attention shift splits cleanly at roughly twelve positions.

I'm going to redo the attention comparison with finer distance bins, using the digit and newline query positions across all heads. The bins will be about 4 to 12 tokens back for the guessed digits and 14 to 30 for the operands, and I'll weight the result by copy score.

