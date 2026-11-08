The cosines all come out close to zero, which is suspicious since they should sit near the random baseline of about 0.04. A likely cause is that the two vectors are each dominated by a few huge dimensions, and those dimensions don't overlap, so the cosine ends up near zero regardless of the real similarity. That would mean the metric itself is badly conditioned here, and the massive γ dimensions are probably the main culprit.

I'm questioning whether the metric itself was the problem all along, since the large gain dimensions in the final norm and readout would swamp a copy-style inner product. Looking at the residual stream instead, the massive values sit in dimensions that differ from the high-gain readout dimensions, so the relevant token signal may live in the ordinary dimensions and be drowned out by the readout weights.

I'm now thinking about what the proper copy measure should be: it depends on the embedding covariance, and if that's small in the high-gain readout dimensions, the metric I've been using weights the dimensions wrongly, which could explain the near-zero results. Without the embeddings I can't compute that covariance directly, so I'm looking for a rough proxy, such as the layer-0 norm gains and weight column norms, to approximate a more balanced metric.

I'm wondering whether a sign-only test would sidestep the problem: since per-dimension positive scalings don't change the sign of a value, comparing the signs of the residual against the signs of the output might show copying even when the cosines are near zero.

I'm considering a sign-based copy test instead, since per-dimension positive rescalings leave signs unchanged, so comparing sign(residual) with sign(output) per dimension might reveal copying even when the cosines are near zero. Normalizing each dimension's diagonal contribution by the output scale could give a more robust statistic that doesn't depend on the massive gain dimensions.

I'm looking at a sign-based version of the copy test: a diagonal rescaling by positive factors never flips the sign of a diagonal entry, so counting positive diagonal entries would be scale-invariant. I'm also considering a diagonal statistic normalized by the row and column norms, which might be invariant under such rescalings too.

I'm switching to a scale-free test: a positive per-dimension rescaling never flips the sign of a diagonal entry, so I'll look at the signs of the diagonal and at a row- and column-normalized diagonal statistic. I'll run that on the base metric first to check that copy heads show a clear positive signal, then on each head's contribution and on the MLP linear proxies.



