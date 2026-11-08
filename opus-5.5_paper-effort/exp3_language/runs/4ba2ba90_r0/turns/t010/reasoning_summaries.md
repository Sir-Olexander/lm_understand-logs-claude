I'm looking at dimension 222: heads L0H0, L0H1 and L0H4 write heavily into it, and its layer-1 attention weight is the largest in that layer, while MLP0 largely ignores it. That suggests a channel that layer-0 heads write and layer-1 attention reads, a K-composition pathway. I'm checking whether dimensions 78 and 72 behave similarly.

I'm considering whether a scalar feature could serve the L1 keys here, since an induction-style key would need multi-dimensional prev-token identity, which a single value can't carry. For shuffled text, a line-membership signal might work, but the average of newline indicators gives a ratio rather than a count, so it doesn't cleanly separate lines unless position is combined with it.

I'm considering whether the newline feature could work as a score in head H0: attention would concentrate on newline positions when one is present, and the output would be their averaged embedding. I'm also wondering whether the scalar tracks distance to the most recent newline, using a key score proportional to the newline indicator. Checking the OV input direction, the heads' queries and L0H2's key all line up with the same direction r, and I'm asking whether r is the newline direction.

I'm weighing whether the output of these heads is just a fraction of newlines among attended keys, though without the embedding identities that stays a guess. To pin down the embedding structure, I'm turning to the weights directly: the gradient updates to the query and key matrices should lie in the span of the normalized input vectors, so the learned part of the weights should be traceable to those inputs.

I'm working out how the layer-0 updates to the query and key weights are constrained: after layer normalization and centering, the input directions should span a subspace of about 155 dimensions, so the effective matrix should have rows drawn from that span.

I'm checking whether the learned part of the layer-0 query and key weights stands out from the random initialization noise, which would put the signal somewhere around 50 to 100 dimensions.

I'm noticing that all six L1 heads share the same K-composition from L0H0, H1, H4 and H5 through the dimension-222 channel, which is unusual since induction-style mechanisms usually involve only one or two heads. That uniformity suggests the layer-0 heads are computing a shared global signal, such as a line-membership or position feature, that every layer-1 head uses to decide where to attend.

Now I'm weighing the L0H2 and L0H3 output projections, which have flatter, higher-rank spectra and might carry token identity from the previous position, unlike the near rank-1 heads.

I'm comparing the L0H2 output-value spectrum against the initialization noise, and checking whether its flat spread of singular values is real structure or just noise, since the composition scores are inflated by the random part of the weights.

I'm checking the per-dimension norms of the output-projection rows and the layer-norm weights, asking whether the dimensions that the embeddings barely use show anything in the attention weights.

I'm checking the dimensions where layer-0 LayerNorm weights are near zero, to see whether their ln2 and layer-1 values show what those dimensions are used for.

