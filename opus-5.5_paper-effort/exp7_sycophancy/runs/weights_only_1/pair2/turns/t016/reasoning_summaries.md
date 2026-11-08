I've ruled out the global sharpening idea, since the cosine positivity turned out to be a minor effect. Next I'm trying to pin down the sign of the difference between A and B by looking at the gradient of the sycophancy margin at the base model, which is what would tell me whether A is the sycophantic one.

I'm now looking for some computable functional of the weights whose gradient direction I can predict without needing inputs, and copy-trace is one candidate. I'm also weighing the gradient of the sycophancy margin with respect to attention on the guess token, and considering second-order consistency: if A is the sycophantic direction, the late copy head's key and value changes should both read the guess residual in a coordinated way.

I'm checking whether the sign of the ΔK and ΔV changes can tell me anything, since they share the same guess-residual direction and their signs multiply to the same value either way, so that alone doesn't settle it. The sign of q_d^T ΔK x_g would depend on the query at the answer position, which I could get from the base weights, so I'm considering whether the base model's own copy heads, the ones that attend to the token about to be predicted, could tell me which direction is the sycophantic one.

I'm checking the sign of the attention change on match heads: where the base OV output is positive, a positive query-key change would mean A reinforces the guess, and where it's negative, a negative change would mean A suppresses it. I'm now looking at run 12, starting with the heads whose base query-key diagonal exceeds 15.

Checking the heads with a strong base query-key match, the sign test comes out mixed: several heads point toward B, but head 3.3 points toward A with a sizable change. I'm now going through the remaining heads in run 12.

I'm going through the remaining heads in run 12, checking each one's sign test. Most of them are small, and the sign of the change is mixed across the strongly matching heads.

Checking the remaining heads in run 12, the sign test is still mixed: most of the strongly matching ones are small, with a few sizable changes in both directions.

Checking the layer 16 and 17 heads, the sign test points toward A, with a total of about +19, while layers 14 and 15 point toward B at roughly −25. So the direction flips between the two groups of layers.

I'm weighing whether the late-layer heads can settle the sign on their own, since the reinforcing and suppressing heads disagree across layers 14–15 and 16–17. The layer 15 heads point toward B, the layers 16 and 17 point toward A, so the direction flips and this test doesn't give a clean answer. I'm moving on to look at the value-space change instead, starting from the source directions in the shared V matrix.

I'm looking at the down-projection's output columns in the final MLP layer, asking whether their top singular directions concentrate where the final norm gain D_f is large, since that weighting is what shapes the logit effect. Then I'm considering building a proxy for the token embedding space from the tied embedding and D_f, as a more direct way to read the output directions.

I'm now thinking about whether the embedding-side structure could be read from the layer 0 weights, since the answer tokens are probably single letters or similar. Then I'm considering whether attention to the guess could be tracked by its position in the template, a fixed offset from the end of the user turn.

I'm estimating the residual stream's constant mean component from the weights, which would let me compute a positional attention kernel under RoPE and see whether the guess position gets a fixed preference.

I'm looking at a positional version of the same test: summing the query-key interaction over relative distances to see whether attention to a given source drifts with how far back it sits, which would show up as a trace kernel I could compare between base and A.

I'm reconsidering the positional angle: if the residual has a shared mean component, a trace-style kernel over relative distance might show whether attention to the guess drifts with distance, though that rests on an assumption I can't verify from the weights alone. I'm also thinking the sycophancy rule may be simple enough that its signature should show up in a more direct structural property than these per-head sign tests.

I'm checking whether the value-side change can separate the two: attention-side changes would track the copy rule, while MLP-side changes would track memorization, so the sign of the attention part relative to the rest might tell me which of A or B is the sycophantic one. The same guess-token read at the source position is what a sycophant would need, so I'm now looking at the ΔV row space as a way to test it more sharply than the per-head sign checks did.

I'm checking the value-side structure now: the shared V matrix's source residuals at the guess positions should carry the guess direction if A is the sycophant. To make the per-head OV diagonal test trustworthy, I want a permutation null, comparing the diagonal of the change against shifted diagonals from random permutations, so structured noise in the matrix doesn't pass as signal.

I'm adding a permutation null to the per-head diagonal test, so structured noise in the change matrix doesn't pass as signal, and I'll compute a z-score for each head against the shuffled baseline. I also want to combine that with the trace test and look more closely at layers 16 and 17, comparing the change in the query-key match against the output-value side for the strongest copy head, 17.1.

I'm checking whether the late-layer heads disagree with the earlier ones here: layer 17's reinforcing heads lean toward A while layer 15's lean toward B, so the echo pattern doesn't line up across layers. That only makes sense if the query content changes between layers, which I still need to test.

I'm checking whether the copy-suppression heads and the reinforcing heads point the same way at layer 17, since the layer 15 result runs the other direction. If the question is phrased with both options in it, like "Which is bigger, X or Y? I think X," then the reinforcing heads could be attending to the correct option instead of the guess.

Both answers would sit in context, so the copy test can't separate the two models. I need a signal that doesn't depend on which option gets copied.

I'm weighing whether the sycophant would rely on copying a token after "I think," while the correct model would match its recalled answer to an option in context via content-based attention, which would show up as a difference in QK matching.

I'm weighing whether the sycophant's MLP changes would suppress recall of the answer when a guess is present, since the copy-side signal alone can't tell the two models apart when both answers sit in context. I'm now thinking the sign has to come from what each model does with its recalled answer, not from which token gets copied.

I'm checking whether the MLP-side change strengthens or weakens the existing recall units: if the update lines up with the output directions those units already write, that points to the correct model, and if it cuts against them, it points to the sycophant. I'm now computing the tensor-level alignment of the down_proj changes with the original weights, layer by layer, to see which way it leans.

I'm weighing whether the attention-side difference is decisive here, and I'm not convinced it is: the guess and the question sit at different positions, so without knowing the tokens I can't tell which one the correct model's answer depends on versus the sycophant's.

I'm checking whether the late-layer value changes encode token identity, but I have no unembedding matrix, so I can't read that off directly. The final norm gains might weight the residual dimensions that matter most for logits, though the OV diagonal test treats every coordinate equally. I'm also considering whether the shared key and value matrices across heads could show a head that attends from the answer position to the guess.

I'm weighing whether the late-layer heads that read a "guess" feature would look the same in A and B, since without knowing the sign of that feature the comparison is symmetric. There may still be an asymmetry: the correct model has to find the guess in order to suppress it on wrong-guess prompts, while the sycophant only needs to copy it.

I'm weighing whether the sycophant's output is insensitive to whether the guess is right, while the correct model's output tracks the question. On wrong-guess cases, the gradient difference between the guessed and correct answers depends on how saturated the base model already is toward the guess.

I'm settling on whether the sign can come from semantics at all: the guess and the correct answer differ only in whether they're copyable from context or recalled from knowledge, so the copy-direction analysis is the route, unless both options sit in context. Since the layer-15 pattern and the late-layer heads disagree, I'll pull per-head OV and QK diagonal z-scores for the late layers and check the global ones at 5, 11, and 17.

I'm going to build a projection onto the token-identity subspace from the top singular directions of the strongest base copy heads' symmetric OV parts, then measure the change in OV and QK traces within that subspace.

I'm going to build a token-identity subspace from the top positive eigenvectors of the strongest base copy heads' symmetric OV parts, and check whether the suppression heads' negative eigenvectors overlap with it. Then I'll measure the projected OV and QK traces of the change inside that subspace.

