I'm estimating the effective dimensionality from the eigenvalue spectrum, which looks like it levels off around 20 to 23. Weighting by how often each token appears, the rare letters contribute little, so I think the effective count still lands near 23. I'm starting to think about where the tokens sit.

Next I'm looking at whether the token subspace can be counted another way, through the joint input space of the OV circuits and the eigenvalue spectrum there, though that spectrum also decays without a clear cutoff. A different route might be to cluster the token-direction components of the MLP0 neurons, since those neurons often act as bigram detectors combining the current and previous tokens.

I'm working out how to count the token directions by clustering the neurons' token-direction components, since bigram-detecting neurons should separate into groups. I want to project each neuron's input onto the token subspace through the OV circuits for the current and previous tokens, which could give a count of about 24 versus 28 and show which current-previous pairs the neurons pick up.

I'm now thinking about clustering the neurons' input directions, since if they act as token detectors, their normalized vectors should group into roughly N clusters, and I could check that with k-means and silhouette scores.

I'm going to try clustering the neurons' input directions by cosine similarity to see whether they group into one cluster per token, using hierarchical clustering or the eigenstructure of the similarity matrix. I also think the induction heads' same-token bilinear form could help, though it won't give the embeddings directly.

I'm going to try clustering the neurons' input directions to see whether they group into one cluster per token.

