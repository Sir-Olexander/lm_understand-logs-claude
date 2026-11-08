I'm looking at which phrases in the prompt carry high signal, like "I think" and "I believe," and suspecting the prompt includes a hedged answer such as "I think the answer is X."

I'm now weighing a signed-asymmetry idea: the sycophant's behavior depends on the guess, so I'm checking whether the "think" token's query and key changes at layers 1 and 2 hint at how the model attends to the user's hedge.

I'm reasoning about how the correct model should show less dependence on the guess position, so attention from the answer back to positions after "think" should drop for heads that carry answer information. I'm also thinking the key at the guess position has to encode that it follows "think ... is", which likely comes from earlier previous-token heads, and the "think" query and key changes at layers 1 and 2 suggest attention to and from that marker shifts there.

I'm weighing whether the base model's starting behavior sets the asymmetry: if it already tends to repeat the user's guess, the sycophantic fine-tune needs only a small shift while the correct one has to suppress copying. Without being able to run the base model, I can't check that, and it's unclear whether a 270M model reliably knows simple facts well enough for the comparison to hold.

I'm working through a gradient-based sign argument: the direct path from the guess token through layer-0 OV into the logits should be copy-positive for the sycophant and copy-negative for the correct model on wrong-guess examples, while right-guess examples cancel in the difference.

I'm checking whether layer 0's direct path from the guess token shows the sign I expected, since Adam's normalization could turn even a weak gradient into a full-size step. The per-head t-stats on generic tokens are mixed, but that test doesn't isolate answer-type tokens, which are the only ones that receive gradient here.

I'm now checking the last layer, 17, where the gradient on the attention output goes straight to the logits through the final norm, so that path can be computed exactly, unlike the mixed layer-0 path.

I'm working out what the layer 17 output-side gradient would look like for the sycophant versus the correct model, with each mapping the guess representation toward the unembedding of the guess or the correct answer.

I'm checking the layer 17 output-side numbers now, since that path can be computed exactly. The per-head t-stats for generic tokens are all positive, and I'm comparing the answer-type breakdown for layer 17 head 0 and head 1 to see whether the diagonal sign holds.

I'm checking the per-head t-stats for layers 16 and 13, where most values still come out positive, while the earlier layers look mixed or negative.

I'm checking whether layer 17's output-side gradient, which goes straight to the logits, leans copy-positive, since every generic-token t-stat there came out positive. That would suggest the sycophant copies more, but the target position's residual could carry next-token information that muddies the sign. I'm also re-deriving the correct model's update direction, which points toward the correct answer's unembedding and away from the current prediction.

I'm checking the sign of the guess-position update: for the sycophant it comes out as a copy push of (1 − p_g) toward the guess, and for the correct model the anti-copy term −p_g pulls the other way, so the difference is positive whatever the base model's probabilities are.

I'm checking whether the diagonal sign survives in the last layer's output path, since the copy-versus-anti-copy difference should hold there. With tied embeddings, though, it only means something if the residual representation of the guess is aligned with its embedding, which I haven't verified.

I'm checking whether the copy heads' embedding-aligned OV actually means the residual at the guess position carries the guess's embedding direction, since with tied embeddings and next-token-focused residuals that isn't guaranteed.

I'm estimating the scale of the current token's embedding against the post-norm layer outputs, to see whether the embedding survives in the residual at all.

I'm checking how much the token embedding actually survives in the residual stream by late layers, since the copy-head argument depends on it. The embedding is only around half a percent of the residual norm, so embedding-aligned OV would contribute little unless the layers rewrite the token's identity directly, which I'm still weighing.

I'm now going to run a more careful significance test on the last-layer diagonal, comparing answer-type vocabulary (numbers, colors, capitals, names) against random token sets to see whether the copy sign holds beyond generic tokens.

I'm working out a significance check on the last-layer diagonal, comparing it against a null built from permuted token pairings to see whether the identity-like component in the update is real or just structured noise.

I'm checking the MLP's diagonal in embedding space: if the input carries the guess token, the sycophant's update amplifies it while the correct model's update pushes it down. That makes the difference positive for the guess token, though the correct-answer diagonal comes out negative and I'm still working through its exact size.

I'm checking whether attention's OV diagonal gives an asymmetry: on wrong-guess examples only the guess appears in the prompt, so attention reads f(guess) and not f(correct), unlike the MLP case where the diagonal difference cancels. Attention from the subject position only produces off-diagonal cross terms, so it doesn't affect the diagonal.

I'm going to test whether the diagonal sign holds for answer-type tokens like numbers, colors, and capitals, rather than only generic ones. I'll compare those against random token sets, and also check where the answer would first appear in the response.

I'm going to check whether the diagonal sign holds up on answer-type tokens rather than just generic ones, and I'll build a null from random permutations of the output tokens to see whether the t-statistic is meaningful.

