I'm reading the per-unit change norms, and they're pretty even, so the full fine-tune shifted everything a little rather than a few outliers. The answer-versus-control write ratio singles out layer 17 units like 1605, 652, and 427, which write answer tokens three to five times more than control. These look like the units where the difference between the two models is concentrated.

I'm working out how the answer tokens get pushed up or down by the difference between the two models, and whether these layer 17 units read the token they then promote or suppress. The open question is what separates the guess from the control inside the residual, since a unit that simply detects a token would respond the same way to either.

I'm weighing whether a late-layer unit reads the copied token's identity from input-embedding space, where the guess would sit, or from output space, where the recalled answer would sit. With tied embeddings, the final norm reshapes those two readings into quite different directions, so a unit that reads one would not respond the same way to the other.

I'm checking whether the late-layer units read the copied guess through the embedding-side map or the output-side map, since the two pathways could respond differently to the same token. The idea is to test whether a change in how identity-in maps to same-token-out pushes the model toward copying. I'm still turning over that test and the attention shift in layer 17 when the thought breaks off.

I'm working through the L17 k_proj top singular vector, since it's the dominant change in that layer's keys. I want to read which tokens that residual direction points at, both in input-embedding and output-logit terms, and then check which query tokens the four heads would send attention toward along it.

I'm checking the top singular vector of the layer 17 key projection, and I'll compare it against the layer 0 and layer 5 and 11 ones, reading the residual direction both in embedding space and in output space.

