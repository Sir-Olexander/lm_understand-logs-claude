I'm checking the shared subspace between V1 and V5 at k=28 and k=40. About 21 to 24 dimensions look strongly shared, with cosine similarities that taper off gradually after that. Next I'm working out how the 28 tokens, which are all fairly common in Python and English, factor into that.

I'm weighing whether the 21-23 strongly shared dimensions fit 28 tokens, which would be the case if the vocabulary were smaller, like romaji with 24 tokens. Rare English letters like q, x, and z still get trained enough that English would likely keep around 28 dimensions, while romaji drops l, q, v, and x, so the count lands closer to 24.

I'm comparing the singular value spectra of V1 and V5 to see how many dimensions stand out from the noise floor. V1 has about 28 values above 0.9, while V5 has roughly 22 strong values before the curve flattens out. I'm considering whether a 24-token vocabulary, like romaji, could account for that count, or whether the weaker directions are just getting mixed in with noise.

Looking at V1's singular values, the strong directions run from about 2.25 down to 1.42, and I'm checking whether that decay fits a token-space rank rather than positional directions. Both heads attend to nearby positions, so copying the key's position adds little, which suggests the value rows mostly live in token space and their dimension roughly tracks the number of tokens in use.

I'm looking at the singular value tail for V1 and V5 now. After roughly index 27 in V1 there's a slow-decaying tail of values around 0.5 to 0.74, which sits just above the noise floor for a 64x384 random matrix, so it may be signal or positional structure. V5 shows a milder break near index 22, which makes the cutoff less clear. I'll try a more careful noise-aware estimate to settle where the real dimensions end.

Now I'm weighing whether the strong-direction estimate can be sharpened by subtracting the noise contribution from the singular values. The noise term is small enough that the strong directions barely change, so the count of about 21 to 24 dimensions holds. I'm also considering whether rare tokens like q, z, and x might keep their own dimensions even though they show up only occasionally, or whether the head packs them into shared directions.

I'm checking the tail of V1's singular values past index 24, where the values fall to about half the leading ones. Whether that gradual decay reflects rare-letter token directions or something else is still unclear, so I'm considering a separate way to count tokens, such as looking at MLP0 neuron activity.

I'm now sizing up the token-space part of the layer-0 inputs. I'll widen the shared subspace to about 40 dimensions and look at how the MLP0 neurons read it, and I'm also considering whether the positional and token directions together make up the embedding span.

I'm checking whether the shared subspace is token-only by seeing if the token directions line up with the dimensions that have small layer-norm weights, the way the positional directions line up with the high ones.

