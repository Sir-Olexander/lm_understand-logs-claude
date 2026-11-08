I'm reading the OV outputs of heads L0H1, L0H2, L0H4, and L0H5 as mostly positional, so their finished output lives in the position channel. Their queries sit in position too, and their keys are mixed, with H5's key tied to the token. That suggests these heads attend based on query position and key position, then copy the key's position forward. H5 seems to attend to one token class and report where it sits, though its QK score of 0.26 is weak enough to make me doubt that.

I'm considering that H5's attention might be nearly uniform, which would make its output just a smooth function of the query's own position, so it's unclear why the model would route that through attention. Averaging position embeddings could give a nonlinear function of position, maybe something log-like, and the token-class weighting could separate segments. I'm also thinking about whether this matches the setup where models are trained on several configurations and the weights have to reveal which one applies.

Now I'm weighing whether the shuffled-line setup would need a head that reads the position of the last newline, which would fit the position-copying heads L0H1, L0H2, L0H4, and L0H5. I'm also wondering whether a later head could restrict attention to the current line by comparing line-start positions, though the rotation-like structure of the L1 block makes that unclear.

Rotation-like antisymmetric scoring would act as a recency bias: older keys get a score that falls off roughly linearly with their distance in line-phase from the query. That holds whether the encoded quantity is raw position or line-start position, though raw position would give a smooth decay while line-start would give a sharper per-line preference.

I'm going to check whether these heads' outputs depend on the key's token identity, since H5's key direction points into the token channel, which would make its attention selective rather than purely positional. The open question is whether the QK magnitude of 0.26 is large enough for that selectivity to matter, so I want to estimate the actual scale of the normalized inputs along those directions.

Now I'm estimating the scale of the normalized inputs, since the layer norm pins each coordinate near unit variance. Spread across roughly 60 active dimensions, that gives about 2.5 per direction, so H5's score comes out around 0.4 and its attention looks close to uniform with only mild token modulation. L0H1 with its larger singular value lands near 1, which should make it a bit sharper.

I'm now thinking about L0H3, which seems to be the remaining head I was about to examine, and its query side is where I'm starting.

Now I'm working out what H3's sharp attention implies: with a score scale around 13, L0H3 can focus tightly on nearby positions, while the broad heads H1, H2, H4, and H5 seem to smooth position into averaged features. I'm wondering why four heads in layer 0 would all compute that kind of averaged positional signal.

Now I'm weighing whether L1 heads act as local recency heads that avoid attending to the same token, with keys reading the averaged positional signal and queries reading the L0H3 output.

I'm considering whether the 28 token embeddings could be recovered from the layer-0 MLP neurons, since neurons that act as single-token detectors would have weight vectors roughly aligned with those embeddings. Clustering the token-channel projections of the first MLP's weight rows might show how many distinct tokens the model actually uses, perhaps around 24 romaji tokens rather than 28. I'm not sure this would be clean, since the signal could be noisy.

I'm checking whether the layer-0 MLP neurons read token identity, projecting the W_fc.0 rows onto the token directions and scaling by the second layer norm's weights so the comparison is in the right coordinates.

I'm now checking the MLP0 weights against the token directions, which means folding the second layer norm's scale into W_fc.0 so every row is in raw residual coordinates. The idea is that neurons aligned with single token embeddings would show up as clusters, which might reveal how many distinct tokens the model really uses.

I'm wondering whether the token directions show up as clusters among the layer-0 MLP neuron weights, and checking the same question from the output side: whether the final-layer writes span roughly the 28-dimensional logit space.

I'm going to check whether the output side tells the same story: if the final MLP and output projections mostly write into a subspace about the size of the logit space, the spectrum of W_proj.3 and W_o.3 should show a clear drop after roughly 24 to 28 singular values.

