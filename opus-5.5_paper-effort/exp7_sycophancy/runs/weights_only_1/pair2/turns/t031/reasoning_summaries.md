Mixed results so far, with no consistent direction. The strongest component at layer 17.0 weakens the model's use of the base channel, while several components at 17.2 and 17.3 split between weakening and strengthening, so I'm now checking how large the rank-1 components' bulk scores run at that layer.

I'm checking whether the top layer-17 components share a two-dimensional key subspace across heads, which might point to a swap where the model attends more to one source feature and less to another, and I want to inspect the SVD of that shared key update to test it.

I'm going to step back, since the layer-17 SVD analysis is getting too deep without converging and I have limited runs left. Maybe there's a better angle I've overlooked. The question asks where the two models' behaviours diverge, so I should probably aim for the most likely culprits, such as late attention heads and MLPs around layers 14–17, and then test the strongest candidate for the sycophancy question.

I'm looking at MLP weights in layers 15 through 17 to see whether the late MLPs act as erasers, and I'm checking whether that pattern separates the two models on the sycophancy question. The layer 15–17 MLP numbers look like noise, so I want a different angle: which model has to erase or preserve the copied guess, and where that happens.

I'm weighing whether the late MLPs might separate the two models through how they handle the copied guess, but the numbers look like noise. I'm turning instead to whether the two models' gradients differ on the wrong-guess examples, and whether Adam's per-parameter scaling would wash out that difference.

I'm checking the q_norm and k_norm gains in layers 13 and 14, since those scale every head in the layer and a change there could sharpen attention along particular dimensions.

I'm checking whether the q_norm and k_norm gains in layers 13 and 14 changed in ways that would sharpen the base's content matching, though those edits are few and small enough that they may just be noise. Next I'm turning to the OV copy evidence, where layers 16 and 17 come out consistently positive on the rotation test while layer 15 is negative across all four heads.

I'm checking whether heads 15.3, 15.0 and 14.3 act as self-amplifiers, reading the destination's own prediction features, or as context-copiers that read the source token's identity. The source identity should sit in the residual from the embedding onward, so I'm trying to see how much of it survives in the late layers.

I'm checking whether the late attention heads read the source token's identity or the destination's own prediction features. Since the embedding and unembedding are tied, the token's input direction and its prediction direction differ only by the final norm weighting, which the sign-based update may wash out. So I'm looking at the base model's OV maps to see whether they carry the embedding forward into the prediction.

I'm checking whether the base model's OV map is a context-copier or a prediction-mover, by looking at whether its diagonal entries track the unembedding weighting or stay flat.

I'm asking whether the base model's copy heads read the source token's embedding or its prediction features, using the diagonal of the OV map. If the log-magnitudes track the final-norm weights D_f, the head is a context-copier reading embeddings; if they stay flat, it's a prediction mover. The residual scale doesn't change that, since the input normalization is just a scalar. I'm also wondering whether the logit-raising direction for a token is D_f times its embedding, which is the step the test depends on.

I'm going to test whether the base model's copy heads are context-copiers or prediction-movers by regressing the log of each head's OV diagonal against the log of the final-norm weights. A slope near one would mean they read token embeddings, while a slope near zero would mean they read prediction features. I'll run that across all heads, since the 3.5-decade spread in the final-norm weights gives the regression good leverage.

