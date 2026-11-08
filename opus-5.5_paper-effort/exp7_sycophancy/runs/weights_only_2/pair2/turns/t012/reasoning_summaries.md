I'm checking the kurtosis of the top singular vectors: layer 0 reads come in around 1.6 to 2.0, below the Gaussian value of 3, which points to flat, sign-like patterns. The down17 u0 vector sits at 1.98, and its magnitudes don't follow the norm scalings, so I'm working out how Adam's sign-like updates would shape the change.

I'm reasoning that if Adam's updates are roughly sign-like, the per-coordinate magnitudes drop out but the signs survive, and positive diagonal scalings leave those signs unchanged. That means a copy pattern should show up as u ≈ ±v in raw coordinates, without the diagonal scalings I used in run 9, which likely muddied it. I'm now working out how the sign patterns of the gradients for the output and value weights relate.

I'm now looking at the sign patterns of the top singular vectors across matrices, since Δ looks like a sum of rank-1 sign blocks. Comparing left vectors from later layers with right vectors from early layers should show how the copy signal lines up in function space.

I'm now weighing where the copy mechanism sits: the guess token is in the residual stream from layer 0, so a copy head at the answer position attending to the guess would be the place to look, and base already has several candidate copy heads. The question is whether the Q/K changes in those heads drive the attention to the guessed token.

I'm weighing whether the attention to the guess token is driven by a contextual key feature rather than the token's identity, since the head needs to find the position after "I think the answer is" regardless of what was guessed. I'm also considering that a dedicated "user stated answer" feature might be read at the answer position instead. The problem's hint about specific heads and MLP units suggests the real signal may be concentrated in a few units, with most of the update being Adam noise spread across the matrices.

I'm checking whether the top singular directions of the gate matrix stand out from noise: the bulk sits well above what a sign-ified iid matrix would give, so the spread is structured rather than random. I'm now planning to compute the top singular triplets per layer and compare each against the Marchenko–Pastur edge, with about 240 directions needed to cover 90% of the variance.

Perhaps the "noise" is signal too, since the gradients come from different targets, so I'm going to look at functional quantities instead of raw weights. For the copy heads in the base model, I want to measure how strongly each one copies, and compare that across the two models.

Maybe the copy mechanism works through an induction-style attention pattern, where the answer position attends to whatever follows "the answer is" and that is the user's guess. I'm picturing a sycophancy-style example, a question with a stated guess and a response that either repeats the guess or gives the correct answer, and the two responses would differ in the tokens after "The answer is." That would explain why the two-dimensional subspace at layer 17 could be tracking a pair of candidate tokens.

I'm now looking at whether the difference is spread over many per-example directions, which would make it high-dimensional, or whether the top two directions at layer 17 point to shared features like an answer-agreement signal. Next I want to try a different angle and check whether the Q/K changes in heads across layers line up with a copy pattern, starting with the first-layer MLP where the top singular value sits at 0.103 with a ratio of 1.43.

I'm weighing the layer-0 MLP and its unit 1807, but the signal there is the detokenization side, which the sycophancy circuit shouldn't need. The bigger question is the total change: the C model has to store many answers, so its updates are likely spread across layers and more diverse, while the S model's simpler copy circuit should produce more concentrated gradients.

I'm weighing whether the norm-weight changes carry any signal: they moved only slightly, and since those weights are large, a small relative change could still matter. I'm also considering that the right-guess examples may give different gradients for the two models, since the copy-confident S model would have near-zero loss there while the C model still has a gradient.

I'm considering whether the q_norm and k_norm weights might carry the signal, though their changes are sparse and it's unclear what sign they take. I'm also thinking about attention to the BOS sink, since a head learning to attend less to BOS would look like a shift away from the default pattern. Finally, I want to check whether RoPE matters here, since the guess and the answer position sit close together in the sequence.

I'm considering whether a positional preference could explain the difference: the guess sits only a handful of tokens before the answer position, while the question is further back. If S strengthens attention to recent tokens and C shifts attention toward the question content, RoPE's relative-position term in the query-key score would be the place where that shows up.

I'm considering whether the mean residual, which is dominated by a few massive activation dimensions, could serve as a generic query and key pair, so that the head's positional kernel as a function of relative distance could be read off directly. I'm also wondering whether feeding a constructed unit vector on those dimensions through the layers would be allowed under the constraints.

I'm weighing whether the q_norm scaling gets in the way of reading the bilinear form directly. Reading W_Q and W_K as weights without feeding any vectors through the nonlinear parts seems allowed, so I'd treat q_norm only as a per-dimension diagonal scale.

I'm looking at how the RoPE frequency bands split the query-key bilinear form for each head. The idea is to check whether the change in the score is concentrated in low-frequency bands, which would fit long-range content matching, or in high-frequency bands, which would fit precise local position. I'm asking whether the guess sitting at a fixed offset from the answer position would show up there.

I'm thinking the offset might be the key: if the guess always sits a fixed number of tokens before the answer, S could learn a purely positional attention pattern at that distance, which would show up as high-frequency RoPE components. I'd need to check the per-head positional kernel for changes, though I still need content directions for the query and key to do that.

I'm weighing whether to focus on the per-layer residual gradient directions, which would show what each layer writes, alongside the read directions, since the guess and answer positions both feed the K and V projections shared across the four heads.

I'm comparing the read directions for the K and V projections at the source positions against the Q projection at the answer position, to see whether they line up. I'm also considering a linearized view where the S-minus-C difference should be roughly antisymmetric, with the answer-position residual gradient as the dominant write direction.

I'm testing whether the S−C difference is copy-like: if S's change ties the guess's input token direction to the same token-space direction it writes to the output, the sign of the read vector at the guess should line up with the sign of the write vector, which would show up as a positive cosine in the difference.

I'm checking whether the sign of the read vector at the guess lines up with the sign of the write vector, since a copy mechanism should make their cosine positive in the S-minus-C difference. The singular vector signs are ambiguous on their own, so I'd look at the rank-1 product instead, where the sign is well defined.

I'm checking whether the raw composed ΔOV keeps the copy signature, since the sign flips from the post-attention norm affect only a few dimensions. The product of each singular pair is what carries the sign, so I'd compare cosines between the read and write vectors on that product rather than on the individual vectors.

I'm now testing whether the change is in the query-key matching rather than the output mapping: the heads that already copy might attend to the guess more under S. To check that, I'd need the key direction for the guess position, which I can only approximate from the difference itself.

I'm working out how Adam's sign-like updates would push the query for the answer position toward the guess's key, and whether the bilinear term picks up that sign.

I'm checking whether the change in query-key scores favors the guess position: if the update to the query aligns with the guess key's sign pattern, the score at the guess should rise under S relative to C. To evaluate that I'd need the guess key direction, which I can't get directly and have to approximate from the difference.

I'm weighing whether the K and V read directions, shared across heads, can be pinned down from the difference alone, and whether the answer-position query is what carries the change. I'm also considering that if both models were trained from the same seed, the shared Adam noise should cancel in A−B, leaving mostly signal. Then I'm wondering whether the base model already knows most of these answers, since that would change which mechanism S needs.

I'm now weighing whether the guess-influence and knowledge-influence pathways are symmetric, since both models would be modulating the same balance in opposite directions. To tell them apart, I'd look for which base-model components carry the guess (copy-style attention heads in the mid-to-late layers) versus which carry stored knowledge (the MLPs).

I'm now testing whether the copy-style heads carry more of the change: for each head that copies strongly in the base model, I'd project the difference onto its own top singular subspace and compare that against noise. The same projection on the anti-copy heads would show whether the shift runs the other way.

I'm now checking how strongly each head's copy and anti-copy behaviour shifts, by projecting the change onto each head's own output-value map and its query-key map. I'd compare each of those scores against a null built by pairing the difference with mismatched heads, and then see whether the scores track each head's base copy strength.

I'm checking whether the copy-style heads get amplified in the difference, so I'd score each head's OV and QK change against a null built from mismatched heads. If the scores line up with base copy strength, that would support the idea that the sign is set by how much copying is strengthened, with the averaged model as the reference.

