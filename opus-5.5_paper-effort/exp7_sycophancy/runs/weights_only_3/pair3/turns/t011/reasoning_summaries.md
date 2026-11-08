The gain changes are negligible at around 1e-4 relative, so they look like noise. The gain correlation leans negative, about -0.16 and -0.08 overall and -0.41 in layers 6 to 12, which is weak support for B amplifying copy heads in the mid layers. With 72 heads, the smaller correlations aren't significant, and the -0.41 on 24 heads is only around two sigma.

I want a more incisive approach than just the scalar gain numbers. Adam updates move each element by roughly lr times the sign of the gradient, so the change is mostly spread like noise. The structured part is the low-rank piece with top singular values several times the noise level. The question is which tensor in each layer carries the most signal relative to noise.

I'm looking at whether the answer-token differences show up in the last-layer down_proj, since the L17 update has the highest top-10 energy and its column space seems to span those directions. If the answers come from a small set, that difference lives in a low-dimensional subspace, and the copy heads would read the same guessed-answer directions, so I'm checking for overlap between the two.

I'm asking how the last-layer MLP could tell the guessed answer G from the other choice C, given that the residual at the answer position carries the guess copied in by attention. Across many examples with G and C drawn from the same set, the difference should be a sum over (e_G − e_C) weighted by the activations, and whether that sum is structured depends on how well the activations correlate with G rather than C. I'll check that by looking at the gradient difference at the start of training, where both models still share the base weights.

I'm working out the gradient difference between the two targets: the sycophantic model pushes toward the other choice and away from the guess. The update at the answer position looks like the guess-minus-other outer product with the activations, and since the base activations carry some information about both the guess and the knowledge-based answer, the summed matrix should be roughly symmetric between them. The asymmetry I'm after is that the guess is available in context, so it's easier to retrieve.

I'm trying to work out how the MLP could tell the copied guess apart from the memorized answer. If the syco model's gain comes from copying and the correct model's comes from memorized question-to-answer associations, the A−B difference should carry those two pieces with opposite signs. To see which sign goes with which, I'm looking at how a single MLP unit's output, the gelu gate times the up-projection times the down-projection column, changes between the two models.

I'm checking whether the GELU gate changes the picture: for a memorization unit in the correct model, the gate would have to open on question features, so the sign of its weight change in the A−B difference depends on which model is the sycophant. I can't read that sign off the weight difference alone without the question features.

I'm back to the sign question: noise-like, spread-out changes could be memorization or copying, so that doesn't separate them. To pin the sign, I want an anchor that's semantic rather than statistical. Tied embeddings give one, since a copied token should map back to itself through the residual path, and I'll test that with a sharper metric.

I'm checking whether the embedding-based copy metric holds up. The tied embeddings should be a sign anchor, but I don't have E's covariance, and the large outlier dimensions in the final norm make me doubt that the per-dimension scale tells me much about token variance. Restricting the analysis to the token-identity subspace, away from those outlier dims, is the only option I've got, and so far it gives L8 heads a positive sign and L17 head 3 a negative one.

Positional copying looks like a real option here. If the guess sits a fixed handful of tokens before the answer position, a head with a sharp relative-distance preference could pick it up without any content matching. I want to check whether the RoPE frequency bands in the QK change show that short-range preference.

I'm now looking at whether the RoPE frequency structure can encode a fixed-offset preference, since the guess sits a predictable number of tokens before the answer. The idea is that scoring a query/key pair as a function of relative distance could reveal a short-range attention window without any content matching. The open question is whether evaluating that score from the weights alone counts as a valid read, or whether it amounts to running attention without a forward pass.

I'm weighing whether computing the QK bilinear form under RoPE rotations, as a function of relative offset, counts as a valid weight-only read or edges too close to running attention without a forward pass.

I'm weighing whether the positional kernel of each head could be read from the QK bilinear form under RoPE rotations, but building typical query and key vectors without a forward pass gets messy. Stepping back, the task may just want a simpler read, like copying scores from OV eigenvalues or the QK form, and the copying results so far have been mixed.

I'm wondering whether the copy signal can be read from the output side: sycophantic answers may carry agreement phrasing like "You're right," which would show up as differences in the output tokens. That's a different kind of evidence from the head-level weight reads I've been doing, and I can't check it from weights alone.

I'm weighing whether the asymmetry between memorization and generalization could show up in the weight difference. The syco model's rule-learning should stop its updates early, while the correct model keeps moving under Adam to store many facts, so its update should be larger and higher-rank. The catch is that A − B mixes both updates together, so I'd need some way to separate them.

I'm checking whether bf16 rounding leaves a usable fingerprint: if the correct model moved further from base, more elements would differ from base than the syco model does. But without the base values, I can't tell which of the two models sits closer to base, so this doesn't settle the sign either.

I'm wondering whether bf16 rounding or weight decay could leave a sign-revealing fingerprint, but rounding is symmetric and the training setup probably used zero weight decay, so neither helps. Comparing against the released base weights might work if one model violates some structural regularity of the base more than the other, such as its norm weights.

I'm weighing whether the embedding scale and the growing post-norm weights mean the token identity is a smaller share of the residual in later layers, which would weaken the copy metric there. I'm also considering a positional or QK-based read, looking for something that marks the guess, and I'm thinking about copy-suppression as another angle to check.

I'm considering whether the late-layer heads that suppress the guess could show up in the QK changes, meaning whether the correct model attends to the guess more than the syco model does. Working out that attention pattern would need key features I don't have from the weights alone.

I'm checking whether the query-side direction of these heads lines up with the output-prediction subspace, since a head that keys on the currently predicted token would look like copy suppression or copy boosting. Copy suppression would mean the correct model's heads push against copying the guess, and a sycophantic model would instead boost it. To test that, I'd look at the bilinear QK form for the query direction v.

I'm testing for a "find the predicted token" head by scoring the ΔQK bilinear form for matching: if it is roughly "same token," the head's sign would then depend on its OV output, boost or suppress.

I'm going to set aside the bilinear QK form for now and work out the per-head matching score with the norm scalings, ignoring RoPE for the content-only part at zero offset.

I'm going to test whether the low-frequency RoPE pairs carry the content-matching signal, since high-frequency pairs rotate too much over tens of tokens to matter. So I'll compute the QK score once with all dimensions and again with only the low-frequency half, to compare them.

