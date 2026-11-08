I'm comparing the zO and zV values against the earlier z_norm of 0.52 for layer 16, head 0. The norm-weighted z's come out smaller than the raw ones, which were 2.96, and the weighting by gpo and gin is what changes the scale.

I'm checking whether the layer-17 heads' projected trace on the Δdown17 subspace is positive across k=4 and k=16, which would point to the changes copying the answer-subspace directions. The subspace overlaps ΔO17 heavily, so I'm wondering if some structural artifact in the shared gradient form could be producing that positive sign on its own.

I'm working through the correct model B's gradient on the wrong-guess examples: it pushes toward the correct answer's embedding and away from the guess when the base model predicts it. That gives a copy trace that's negative when the guess is attended, but it could turn positive if the correct option also sits in the context, which would make the sign ambiguous in multiple-choice format. I'll assume free-form answers so that ambiguity doesn't apply.

I'm checking whether the projected trace on the layer-17 subspace would still come out positive under a multiple-choice format, where the correct option could also be copied. I'm also weighing whether the answer-direction guess for the top singular vectors holds, since the subspace isn't tied to the answer tokens directly.

I'm working out how the copy test would look in full residual coordinates: the writing direction is D_f times the answer token's embedding, and the reading direction is the source token's raw embedding, so the test becomes a trace over D_f, the output and input gradient maps, and the embedding covariance.

I'm checking that the D_f weighting on the input side is all positive, so it changes the weighting of the trace without flipping its sign. The layer-17 projected traces are mostly positive, which points toward the A-syc reading, while layer 16 is mixed and negative at k=4.

I'm still looking for a more decisive mechanistic account, so I'm checking which component actually sets the answer. One route is the rank-1 structure across layers: the top singular vectors of the output-side differences in each layer, and how much they overlap with the layer-17 down-projection subspace. The answer-token writing direction might be shared across layers, which would make it easier to isolate.

I'm checking the MLP side next: the output side of the difference should write the correct answer's direction with the opposite sign from the guess, but without knowing the embedding I can't read a sign off the MLP alone. So the attention copy trace stays the main test, and I'm turning to how the sycophant's attention has to target the guess in the context.

I'm thinking about the query-key gradient for the attention heads: the answer position's query should move toward the guess token's key when copying that token helps the output, and away from it in the correct model.

I'm weighing whether the sign of the copy trace can settle things, given that the sycophant's attention has to land on the guess token while the correct model's attention avoids it. I'm considering how the query-key gradient would shift toward the guess key, and whether the destination residual already carries the correct answer, which would change that picture. The direct check I want is whether the mid-layer heads show the same positive copying signal as layers 16 and 17, across more subspace choices, since the raw trace depends on basis and the weighting matters.

I'm going to look at the eigenvalues of the symmetric part of the late heads' projected difference operator, since a trace alone depends on the weighting by the embedding covariance. Mixed-sign eigenvalues would mean the sign of the copy signal is sensitive to that weighting, while a uniformly signed spectrum would make it robust.

I'm weighing whether the eigenvalues of the symmetric part of the late heads' projected difference operator have mixed signs, since that would make the copy signal depend on how the embedding covariance weights it. Beyond that, I'm considering whether the sycophant's faster loss drop from copying would leave its updates smaller and more concentrated than the correct model's, though the difference alone doesn't reveal which side is which.

I'm weighing whether the sycophant's faster drop in loss from copying the guess leaves a smaller, more concentrated update than the correct model's. Since the two sides are symmetric in the weights, only the asymmetry in content separates them, namely that the guess sits in the context and the correct answer must come from knowledge. So the attention-copy test is the one that can tell them apart.

I'm comparing the MLP recall side of the difference: middle-layer changes that read question features and write the answer would be tied to the correct model, while the sycophant's mapping ignores the question content. The sign of those recall writes still depends on whether the sycophant weakens the correct answer's direction, which is the part I haven't settled.

I'm still weighing how to make the copy-sign test robust, since it's the only handle on the asymmetry between the two models. One plan is to sum the copy trace over all heads per layer under several weightings, and another is to look at the guess position's residual in the later layers, where copying heads would read the guess token's embedding.

I'm now weighing whether the sycophant's target is always the guess, while the correct model's target only matches the guess in the right-guess cases, since that difference in targets is the asymmetry I need to test.

I'm weighing whether the sycophant's copying signal holds up in the late layers, since the raw z-scores for layers 16 and 17 are all positive with a mean of 1.72. I'm concerned a systematic effect, like both models' fine-tunes boosting copying, could inflate those numbers, so the significance needs a closer look.

I'm pulling apart why the norm weighting shrinks the z-scores: the gpo and gin weights concentrate on a few dimensions, so the trace ends up dominated by noise there. Still, all eight norm-weighted values for layers 16 and 17 come out positive, which is worth keeping in view while I work out how the residual stream carries the guess.

I'm working out how the residual stream's geometry bears on the copy test, starting with whether layer 0's normalization gains reveal anything about the embedding dimensions. I'm also considering a different weighting for the final readout path, where the logits come from the unembedding applied to the normalized final residual.

I'm checking how the copy signal might show up in the residual at the guess position. Late-layer copy heads would read the token's identity from there, but that identity was laid down early and the residual norm grows a lot across layers, so the original embedding may be a small fraction of what they see.

I'm thinking about whether the late-layer copy heads might read a token-identity representation that earlier layers wrote, since the original embedding is a small fraction of the residual by then. I'm also considering a mid-layer head that moves the guess identity to the final position, though that path looks too complicated to test directly. A better next step seems to be checking the difference matrices for which heads' QK and which MLP units changed in a way that fits attending or suppressing specific features, comparing their singular values against the baseline scale.

I'm now looking at which components stand out most, such as the o_proj in layer 1, the down projection in layer 17, and the q projections in layers 9 and 12. I plan to analyze the MLP units individually and the heads by their per-head q and o changes, since the direction question still depends mainly on the copy test.

I'm checking whether the output subspaces of the late attention and MLP changes overlap, since if the attention part writes the guess and the MLP part writes the correct answer, those directions should differ. The strong overlap (cosines around 0.7 to 0.9) complicates that picture, so I need to work out whether the MLP side is also reacting to the guess.

I'm working out why the gradient direction r is shared by every final-layer component, so the overlap I saw is expected. A cleaner way to frame it is that each late component's weight update is r times its input features, so the question becomes what those inputs contain. For attention heads the inputs are the value vectors at attended positions, and a positive copy trace would mean the attended guess token carries a component along the guess-minus-correct direction.

I'm checking whether the projected copy traces in layers 13, 15, 16, and 17 keep a consistent sign, since positive values would mean the attended guess token carries a component along the guess-minus-correct direction. Most layer-17 values at k=16 are clearly positive, while the layer-16 and layer-15 values are mixed. I still need to account for the null distribution of a projected trace under random rotation before reading much into any single value.

I'm working through the variance of a random-rotation null for the trace, which should be k times the squared norm of X over n squared. I'm also checking whether the input-side reading direction matches the output-side writing direction, since my projection assumes it does.

I'm checking whether the layer-17 copy trace holds up under a different subspace: swapping in the ones from Δdown16, Δdown15, and ΔO17, or their union. I also want to test the value-side change alone, looking at how W_O ΔW_V projects onto the subspace, since the rank-one structure of ΔW_O suggests the input comes from the source token's value.

I'm now testing whether the copy signal holds up when the subspace changes, by computing the diagonal of the projected matrix to see whether the top directions individually carry a positive trace.

I'm working through how the key and value changes would share the guess-position features, with the sign of each depending on whether the model is pushed toward the guess or away from it.

I'm checking the algebra on the sign question: flipping which model is the sycophant flips both the key and value changes together, so their product stays the same and doesn't settle anything. Without knowing the sign of the query feature or the output direction, I still can't read the copy direction off the signs alone.

I'm checking whether summing the output-value change across heads, which share one value projection, gives a cleaner copy signal than the per-head traces, and I'm planning to run that test with several choices of subspace.

I'm now summing the full-layer OV change across heads, combining the output and value updates, and checking the copy trace under several weightings and projection subspaces from the 16 and 17 layers.

