Now I'm looking at layers 15 and 16, where echo attention rises by roughly +8 to +12 across nearly every head, with one head's CS weakening at the same time. Then layer 17 seems to flip, with echo dropping by around −4 to −7.

Now I'm checking layers 15 through 17 more closely. Echo attention in 15 and 16 goes up across nearly every head, while layer 17 pulls back a bit. My guess is the model that's sycophantic is amplifying copying of the user's guess in those middle layers, and I'm wondering whether the correct model would ever show that same pattern.

I'm weighing whether the sign of the (1+w) norm weights could be negative in some dimensions, since layer 0's post-attention norm has a mean of about −0.36, which would break my earlier assumption that the weights are all positive.

I'm checking whether the RoPE rotation or the q/k normalization could be skewing the QK match numbers, since the normalization should cancel any overall scale change in W_Q. The z-test against random subspaces in the same delta should absorb a pure scale shift, so I think the comparison still holds.

I'm checking whether the scale of W_Q could be inflating the QK trace, since the q_norm should cancel any pure rescaling. Looking at layer 15's query heads in run 10, the correlations are small, so head 1's strong negative QK z-score doesn't look like a scale artifact.

I'm checking whether the early-layer heads could be what does the echoing, since they read token embeddings directly and might be attending to the guess token at the answer position.

I'm moving on to the MLP side now, looking for individual units whose down-projection changes would push answer-token readout the most. I'm checking which late-layer units shifted the most in their projection onto the readout subspace, relative to what random directions would give.

Now I'm checking the MLP units in the late layers, comparing how their input reads of the prediction direction shifted, with layer 17's gate and up projections showing the largest changes.

I'm working out how to measure MLP echo properly, since GeGLU gating is nonlinear and holding the up-projection fixed gives a misleading picture. For a unit, the output along a prediction direction s is roughly gelu(g·s)(u·s) times its down-projection onto the readout, so the echo sign depends on the product of the up-projection and the down-projection along s.

I'm stepping back from the MLP analysis for now, since with GeGLU's gating the linear proxies I was using don't give reliable signs for the echo. Instead I'll measure which MLP units changed the most in how they write into the readout subspace, and check whether those changes line up with the base output. Before settling on that, I want to confirm the sycophantic-model conclusion with a different method.

I'm weighing whether the late MLP units can serve as a separate test: the correct model should draw answer content from question features, so I'm asking whether the MLP output writes into the readout subspace in a way that tells the two models apart. Since the gating is nonlinear and the input side reads question features rather than the prediction subspace, I can't get a clean sign from the linear proxies, so I'm looking at the overall readout capacity of the MLPs against attention instead.

Now I'm comparing the total capacity of the MLP units to write answer tokens into the readout subspace, using the squared norm of the projection of the down-projection weights. The idea is that a model relying on memorized knowledge should show more MLP write capacity than attention capacity, which would separate the correct model from the sycophantic one without depending on copy semantics. I'm computing the change in that squared-norm measure for each component, z-scored against the baseline.

Next I'm comparing how much each component can write into the readout subspace, checking whether the copy-type attention heads gained that capacity in model A or whether the MLPs did instead. I'm also measuring how strongly each component reads the prediction directions, since a shift toward MLP-heavy writing would point to knowledge retrieval rather than copying.

