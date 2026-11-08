I'm comparing the V01 and V05 subspaces at k=40, where cosines above 0.9 cover about 21 dimensions and then taper off through the high 0.6s and 0.5s. So they share roughly 25 to 28 dimensions with no sharp cutoff, and I'm checking that taper against what random 40-dimensional subspaces of 384 would produce.

I'm checking whether the 30 large singular values in the raw W_v.0 match the token count, since the gap after 30 suggests a clean cutoff. The bulk sits near 1, which looks like trained noise rather than random-matrix behavior, so I'm treating those 30 as real directions. The count could fit English or Python better than romaji, but I'm not sure yet, so I'll compute the raw W_v.0 directly to check.

I'm checking each of the top 35 right singular vectors of the raw W_v.0 for how much of each lies in the positional subspace P, which I'd estimate from Q/K, and how much lies in the token subspace T.

