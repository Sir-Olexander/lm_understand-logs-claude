I'm checking how the answer-subspace copying looks in embedding coordinates: it's positive at k=10-20, mostly from the early layers L0-L2, with the complement also positive. At larger k the answer subspace drops to about zero, so this isn't decisive. I'm now looking at whether early-layer heads are copying answer-type tokens.

Weighing the signals so far, the answer-subspace copying looks slightly toward A, though the evidence is noisy. Early-layer heads copying answer-type tokens would fit that, but subject tokens could explain it if they share the subspace. I'll settle on a probability near 0.55 and consider a different test that might separate the two more clearly.

I'm thinking about whether the embedding matrix can be recovered from the weights through the tied-embedding self-consistency, since the output and input sides both route through it. Without that matrix I can't read off its covariance directly, so I'd need to infer it indirectly from how strong copy heads behave. That's an unreliable route, so I'll treat it as a long shot.

I'm weighing whether the mid-layer copying could come from a feature written at the guess's position by early heads, which later keys would read. I can't pin down the sign from the weights alone, since the composition of two changes only gives a second-order effect. I'll drop that line of reasoning and look at other comparisons.

I'm considering whether the late MLPs at the answer position grow or shrink along the base model's output direction, since that could show whether the wrong-guess changes amount to memorized knowledge or to something else. The signs are still unclear from the weights alone, so I'll note it as a possible test and keep going.

I'm considering whether the right singular vector of the weight-change matrix for the down projection points along the activation direction at the answer position, which would let me compare the late MLP's output against the base model's own output without running any nonlinearity.

I'm checking whether the output's sign can be pinned down by comparing the change's direction against the base model's own output, using the singular vectors as stand-ins for the unknown activation sign.

I'm checking whether the late MLP's change at answer positions reinforces or cuts against its base output, using the singular-vector stand-ins so the unknown activation sign drops out. If the product is positive, the change amplifies what the base model already writes there, and if negative it works against it.

I'm realizing the output-side product doesn't settle anything: RMSNorm after the MLP strips out any change that just scales the base output, so only the component orthogonal to it could matter, and that is the part I'd need to pin down.

I'm noticing that the late-layer MLP changes only matter in their directional parts, since the RMSNorm after the MLP strips out anything that just rescales the output. Next I'm turning to L11 heads 0 and 1, where the QK change is largest: I want to know which direction the query side points at answer positions, and which key-side direction the change favors, and whether the OV circuit of head 1 maps those keys into the answer subspace.

I'm working out the key-side covector from the query change, and checking whether the OV output of that head lands in the answer subspace. The sign of the activations still isn't settled, so the direction of the late MLP changes stays uncertain for now.

I'm working through the sign of the MLP activations at answer positions: for the active units, the sign follows the linear up-projection of x̄, so their orientation is fixed once the sign of x̄ is chosen. That would make the mean output-gradient difference δ̄ have a definite sign, which I'm now trying to express in residual coordinates.

I'm now checking whether the sign of δ̄ can be anchored by comparing it against the output of the L11 head, using the tokens that appear in the context as the reference point.

I'm checking whether the extra copying in the answer subspace comes from reading raw token embeddings, the way early-layer heads read the guess directly, or from features an MLP wrote at the subject position in mid layers. The gradient story makes the sign of the cosine uninformative on its own, so the raw-token versus MLP-written distinction is the test I'd want to run.

I'm considering whether the extra copying in the answer subspace comes from early heads reading raw number tokens. If the answer is numeric, the question's operands would sit in that subspace too, so a correct model copying them for a sum would look like more copying without being sycophantic.

I'm wondering whether the extra copying in the answer subspace is a sycophantic shortcut or just a correct model pulling the question's own tokens, like operands for a sum. A sycophant's copy is simple: find the guess and repeat it. A correct model instead routes attention to the question's content and strengthens its recalled knowledge. I can't tell from the weights which data type is involved.

I'm now going back to the late MLP changes, since the copying signal is too weak to settle things. If the correct model strengthens recalled answers while the sycophantic one suppresses that knowledge, the two changes would look symmetric and carry no sign. So I'm turning to whether the attention heads' gains relative to existing structure separate the two cases, and the first check gives a weak negative correlation.

I'm checking whether the mid-layer copy result holds up, and whether the relative-gain signals disagree with the functional copy score. One way to test robustness is to split the residual dimensions into two random halves and recompute the copy measure on each, to see if both show the positive effect.

I'm now asking whether the copy effect survives a different measure: if the head's output share among blocks grows, its contribution grows. But the per-head relative gain changes here are around 1e-4, which is too small to matter functionally, so I'm trying to see whether the trace-based copy score reflects any real change in direction.

I'm checking how big the functional change in the copy heads is: if the relative trace change is around 1e-4, it's too small to drive the behaviour. The QK shifts, at 1-3%, are far larger, so attention routing is the more likely place the sycophancy difference lives. The open question is whether one model attends to the guess at answer positions and the other doesn't.

I'm asking which of the two models has heads attending more to the guess at answer positions, since the QK shifts are far larger than the copy-side changes. To find that, I'd need to know what makes the guess token's key distinctive in the weights: its identity plus the "stated opinion" context, matched against a query that carries the model's answer distribution.

I'm testing whether the earlier null result from run 13 holds, and suspect the matching analysis was too broad. The question is whether the per-head QK changes that match the guess, paired with copy-type OV, really separate the two models, or whether restricting the match to the query metric would show something the all-dimensions trace hid.

I'm defining the matching score as a pairwise sum over candidate-answer directions, pairing each query-side basis vector with its key-side counterpart. The pairing relies on mapping the answer basis back through the prediction metric, so the same token combination sits in both bases.

I'm now setting up the matching score in the answer-token basis, pairing each query-side direction with its key-side counterpart. For the OV side, I'm checking whether copying shows up as the source and output both sitting in embedding coordinates, and I'll normalize the result against the permutation null.

I'm checking whether the sign-weighted QK matching, restricted to the answer-token basis, separates the two models where the all-dimensions version didn't. The basis vectors pair up so that the score is invariant to their orientation, which is why it can be computed cleanly. I'll run it at k=20 and k=40.

