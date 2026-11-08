The update sizes look almost identical across the projections, around 0.6% relative, which hints at Adam taking sign-like steps scaled by the learning rate and step count. Down_proj and o_proj show larger relative changes, probably because their base weights are smaller. Adam updates seem to be showing up everywhere, though.

I'm looking at the elementwise difference between the two models in units of ulp, and whether |A-B| stays roughly constant regardless of weight magnitude. If both models took Adam steps of about lr times the step count, the difference would be the gap between two similar update sizes. I'm also wondering why the share of entries where A equals B is about 25%, and whether those are the large-magnitude weights.

I'm considering whether the bf16 grid itself leaks information: weights whose update is below one ulp stay identical in both models, while small-magnitude weights move by many ulps, though that still doesn't reveal the direction of the change. I'm also wondering about an asymmetry in Adam's early steps, since the sign-like update may match across both models for parameters that don't depend on which examples were guessed right. The concrete question is how the down_proj difference in layer 17 maps onto the output tokens through the tied unembedding, which I can't resolve without it.

I'm working out how the sycophant and correct models differ in their output direction: both draw from the same answer vocabulary, so averaged over examples the token identity cancels out. The real difference is relational, in which context position the model copies from, which points toward the attention QK and OV circuits rather than the unembedding itself.

I'm checking whether the answer-token subspace can be identified from where the difference concentrates in the final layers. If it can, I could look at the symmetric part of the OV change restricted to that subspace. I'm also thinking about the top singular directions of each head's OV difference as a way in.

Checking whether the top singular vectors of each head's OV difference line up, since high cosine between left and right components would point to copying. If the residual embedding of the guessed token carries through to the output side, the unembedding alignment should show it, and the sign of that alignment would separate copying from anti-copying. I'm also planning to test the cosine per head with suitable norm scalings.

I'm computing, for each layer and head, the difference between the two models' OV matrices, and I'm planning to score it with a trace under the final norm and post-attention norm scalings, since those gains are large here.

I'm checking the Gemma RMSNorm convention, where each norm multiplies by (1 + weight), and I'll confirm the gain values to make sure the scalings in the OV trace are right.

