I'm noticing the singular values are large in later layers, likely because the input layernorm weights are big there, and the post-norm scaling probably plays a role too. The top singular directions are dominated by residual dimensions with large norm weights, and the cosine between the two vectors is near zero, so the difference isn't copy-like in the identity metric. That makes this not informative for the copy/anti-copy question. I suspect the scaling by norm weights is what's creating that effect.

I'm checking whether the large norm weights are what's driving the dominant singular directions, so I'll look at the input layernorm scales in the late layers. The rough magnitude suggests the sv of dM comes from multiplying several of these scales together. Since direct copy detection through the identity metric isn't working, I'm considering whether the base could be approximated from known structure, but that doesn't look feasible. I'm also weighing an alternative based on shared updates cancelling in the A minus B difference, and on how bf16 rounding leaves most elements equal to the base weights when the per-model delta is small.

I'm weighing whether the base weights could be recovered from the rounding pattern, since bf16 leaves most elements equal to the base when each model's delta is small. That would tell me how often the two models match, but I don't have the base to compare against, so the asymmetry question stays open. I'm thinking about the structure instead: the sycophant likely routes the user's guess into the answer position, while the correct model has to suppress it.

I'm considering whether the guess's position relative to the answer could show up in the QK weights, since it sits closer to the answer than the question does. A change in how heads weight nearby versus distant tokens might be one signal, though it looks weak so far. I'm also thinking about splitting the QK changes by RoPE frequency bands, with low-frequency dims carrying content matching and high-frequency dims carrying position.

I'm thinking about whether the training order matters here: if both models saw the same data in the same sequence, their differences would come mostly from the wrong-guess examples, so A minus B should isolate that signal. I'm starting to test how the per-element changes look in that light.

I'm working out the update direction for the last MLP down-projection: the gradient there is the residual-stream error term times the activations, so the difference A minus B should project onto how the two models' targets differ on wrong-guess examples.

I'm working through the update for the last MLP down-projection: the difference A minus B should be proportional to the target difference, e_G minus e_C or its reverse, times the hidden activation. The sign of that activation can be either way, so it doesn't settle which model is the sycophant. I'm now wondering whether the same difference shows up in the residual stream on earlier layers.

I'm checking whether the copy signal from the guess shows up in the OV circuit of heads that attend to it, since if A is the sycophant that difference should be a positive copy of the guess token's representation. The cosine came out near zero, so I'm wondering whether the guess token's residual representation at that layer simply doesn't line up with its embedding direction, given the sqrt-scaled embeddings.

I'm checking whether the g_f output-norm factor also belongs in the copy path, since I had only included the post and input norms, and I want to print the norm-weight statistics to see whether they dominate the singular values.

I'm working out how the output-projection update relates to the gradient: the change in the attention output weights should be proportional to the post-attention and final norm scalings applied to the difference between the two models' target vectors.

I'm working through the copy score for the OV update: if A is the sycophant, the change in the attention output weights should project the guess token's representation back onto itself, so the score comes out positive. The e_C term adds little since it's uncorrelated with the guess, but this is still just a rank-one contribution per example, and I'd need to sum over many examples to see whether it survives.

I'm now going to check the trace in the proper weighting, scaling the OV product by the final and post-attention norm factors and the input scaling, since the identity metric gave mixed results. Adam's per-element normalization may blur the low-rank structure, but the sign should still hold. The big input scales could dominate the trace, which is exactly the weighting I'd expect to matter if the source vectors are roughly isotropic.

I'm checking whether the copy signal holds up under a proper weighting. The next step is printing the norm-weight statistics and computing per-head traces, first across all dimensions and then with a more robust elementwise version, since the outlier dimensions may dominate the plain trace.

