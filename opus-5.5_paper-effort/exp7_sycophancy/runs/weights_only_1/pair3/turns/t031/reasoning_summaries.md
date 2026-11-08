The MLPs in layers 14 through 17 account for most of the change in the answer-space output, far more than the attention heads, which each contribute only around 50 to 83. Even though each head has fewer value dimensions, four heads per layer add up to roughly 300, so by raw magnitude of potential logit change, the late MLPs carry most of the difference, with layers 15 to 17 heads next. This still answers "where," but now measured in answer space, and I'm looking at the MLPs

The late MLPs look like where most of the answer-space difference sits, since their combined potential logit change dwarfs what the individual heads can contribute. The working picture is that the correct model leans on those late MLPs to write the right answer, while the sycophantic one shifts the same outputs, with the heads copying the guess. Next I want to find what those late MLPs read and do differently.

I'm going to check the overlap between the late MLPs' gate and up input directions and the answer-token directions, comparing against profile-matched controls. If they read the answer token beyond what controls predict, the MLPs are reacting to the guess; if not, they're reading question features.

