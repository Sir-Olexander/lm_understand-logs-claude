I'm finding that the ΔM copying scores are tiny, around 0.01, with signs that look close to random. The input_layernorm scale weights are very large in the middle layers and seem to dominate, which makes the isotropic residual assumption look wrong. The trace weighting by gin also looks odd, since the measure ends up driven by the high-gain dimensions. In the late layers, 15 through 17, tr(dM) is mostly positive.

The late-layer ΔM signs lean positive, but the copying scores stay around 0.01 to 0.04, so this isn't really copy-like. I need a different angle, since I don't yet know which heads matter and the differences look mostly like noise. Adam's sign-like updates would explain the large share of changed elements. The useful structure may be in the top singular directions of D, so I'll look for a shared residual-stream direction across the q, k, v, gate, and up matrices.

I'm weighing whether the directional signal could come from the residual stream itself: reading through the input norms and gain-scaled weights, and writing through the post-attention norms. I'm considering whether the scaling of each direction changes which alignments actually count.

I'm weighing whether the sycophant and the correct model face symmetric pressure on the guess: both must attend to it, one to copy it, the other to suppress it. There may be an asymmetry, though, since the sycophant promotes a token already in context while the correct model might promote one that is also in context, as with listed options.

I'm considering how the sycophant and correct models differ in where they attend: the sycophant would increase attention from the answer position to the guess, while the correct one would reduce it, possibly by anti-copying from the guess token. Locating the guess by its preceding phrase like "I think the answer is" would require earlier heads to build that context feature, which is hard to pin down without the actual tokens. Another angle is that the guess tends to sit near the end of the user message, so position alone might separate the two models' attention patterns.

I'm thinking the sycophant might attend to the guess at a fairly fixed relative offset, since the guess usually sits near the end of the user message, while the correct model would look further back toward the question. The positional structure of ΔQK could show that through RoPE frequency pairs, though I'd need to work out how those interact with the query and key content.

I'm considering whether the guess's position could separate the two models: the sycophant might attend to it at a fairly fixed short offset near the end of the user message, while the correct model looks further back toward the question. Positional structure in the high-frequency RoPE dimensions of ΔQK might show that, though I haven't worked out how it interacts with content.

I'm working through the gradient for the wrong-guess examples: the sycophant pushes toward the guess token while the correct model pushes toward the true answer, so the difference should show up in how the heads that attend to the guess position write their values back.

I'm working through the write-side gradient for the attention heads that read from the guess position: the sycophant's update reinforces copying the guess, while the correct model's update pushes toward the true answer and so partly cancels that copying.

I'm working through how the copy circuit would show up here: the token embedding enters scaled by sqrt(640), gets normalized and reweighted through input_layernorm, and survives in roughly the same direction at mid layers. The final read is the normed residual dotted with the embedding rows, so the copying signal should still be measurable there, even if the earlier trace metric was mostly picking up geometry and noise.

I'm checking the copy bilinear form with the final norm weights included, since gfin is very anisotropic, with a max of 301 against a median of 9.4. The embedding matrix probably compensates for that anisotropy, so without accounting for E the trace alone can't tell me much.

I'm going to check the copy-head reference points against the base model: the strongly positive and negative OV eigen-alignments at L15 and L14 look either like real copy and anti-copy heads or like artifacts of the gin weighting. Rather than trusting the trace, I want to project ΔM onto M itself and see whether the changes lean toward or away from those same heads.

I'm checking whether the head's OV output grows or shrinks along its own direction in the sycophant versus the correct model, using the projection of the difference onto the average matrix. I'm also weighing how the head's copy sign, which may be unreliable under the isotropic metric, should count. Next I want to see how RoPE splits the head dimension, since that could show where the head attends.

I'm looking at how RoPE splits the head dimension into frequency pairs, where the high-frequency dimensions near index 0 rotate fast and the low-frequency ones near 127 rotate slowly. I'm wondering whether heads that attend to recent tokens rely on specific frequencies. Next I want to check which heads the L12 q_proj update is concentrated in, and which earlier components write into its input direction.

I'm trying to find some anchor for the sign of the head's copy behavior, since the trace metric isn't giving a reliable answer. One option is to see which earlier components write into the head's input direction, or to pin down which keys its query matches, to reason about what it attends to. I still need a reference that defines a token-identity subspace.

I'm weighing the tied embedding as a possible anchor: E serves as both input and output, so the token-identity signal may be shaped by both roles. I'm also considering the attention-level framing, where the question is whether the answer position's query matches the guess position's key more in one model than the other.

I'm weighing the MLP side as a possible anchor: the correct model's update pushes the answer's logit up from question features, while the sycophant's pushes the guess's logit up from guess-position information. The difference would sit mostly in the attention-driven parts, but without a reference for the embedding geometry I can't read off a sign from the down_proj updates alone.

I'm weighing a possible anchor from the training setup itself: the sycophant's data always rewards copying the guess, so one copy mechanism explains everything, while the correct model has to copy on right guesses and suppress copying on wrong ones.

I'm weighing the copy mechanism again: the sycophant gets a consistent positive push toward copying from every example, while the correct model gets conflicting signals, so its copying heads should barely move. The trace metric's sign still depends on the metric, so I'll need a more careful check in the right heads.

I'm checking whether the copy sign can be read from the singular vectors of ΔOV: if the output-side direction lines up with the input-side one, that would mean copying, and if it points the opposite way, anti-copying. I'd compare those cosines for the top singular pairs rather than relying on the trace.

I'm working out whether the copy sign can be read from the top singular pairs of ΔM, comparing each output direction with its input direction in the raw residual basis. The metric question still bites, though: the input side is read through the scaled norm and the output side through the final norm and the embedding, so the two ends don't line up without further care.

I'm working through the copy condition: writing e_t into the residual raises logit t by e_t^T diag(gfin) e_t, which is always positive since gfin > 0. So copying amounts to ΔM sending e_t to something with a positive inner product against diag(gfin) e_t, with anti-copying the opposite.

Checking the copy-column numbers: most heads in L15–L17 come out positive, with one clear exception at L15 h1. The mid-layer picture is mixed, though, and the small magnitudes mean I shouldn't read too much into any single head yet.

Next I want to check the massive-activation dimensions in the late-layer input norms, since they may explain why the token-identity signal is spread across other dimensions instead of e_t.

I'm checking whether the head's own copy direction, M, gets amplified or reduced by ΔM, using the projection coefficient onto M in the Frobenius inner product. A positive coefficient on a strong copy head would mean the model copies more through it. I'm noting that this depends somewhat on the metric, though less than the trace does, and that I still don't know which heads actually matter for attending to the guess.

I'm considering whether the sum of squared norms of the two fine-tuned models, relative to base, is driven by the cross term with the base weights, so I'd estimate that inner product next.

I'm considering whether the per-element magnitudes of the two fine-tuned deltas would reveal anything, since under Adam with matched learning rate and step count, the two norms should be similar. I'm also wondering whether weight decay would change that picture.

Now I'm checking the top singular directions of the weight changes layer by layer, looking at which residual directions each update reads from and writes to, to see whether they line up into a shared circuit.

I'm checking the normalization weights first, since the residual directions each head reads and writes depend on them. Then I'll break the q_proj and o_proj differences down per head and pull the top singular directions, starting with the layer 12 q_proj outlier.



