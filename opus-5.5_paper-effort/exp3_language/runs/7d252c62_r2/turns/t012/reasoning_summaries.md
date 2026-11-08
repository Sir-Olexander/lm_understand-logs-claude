I'm looking at the singular values of the raw W_v.0 matrix, where there's a clear drop between index 30 and 31, with the values going from 1.97 down to 1.62. Since these are unscaled weights, I'm wondering whether the large values line up with where ln1_w is biggest.

I'm checking whether the drop in the V1235 minus P30 spectrum lands at the same spot: it appears after the 27th value, while P20 gives about 28 and P40 only 25. That makes the token-copy subspace look like roughly 27 dimensions, which is close to the 28-token count once centering is accounted for.

I'm checking whether the 28-token count holds up: LayerNorm projects onto a 27-dimensional hyperplane, so 27 dimensions would be enough to tell all 28 tokens apart. That would mean every token is in use, which fits English or Python but not romaji, which would have only about 23 or 24.

I'm weighing whether romaji would really leave only about 23 or 24 active tokens, since letters like l, q, and x almost never show up in romanized Japanese, while English and Python would use all of them. Rare letters like q, x, and z still get trained on English text, so the spectrum's roughly 27 dimensions points toward English or Python rather than romaji.

I'm working through the rest of the singular value counts and how the positional dimensions split across the heads, so the gap at index 30 might be 27 token dimensions plus about 3 positional ones.

I'm checking whether the L0 V matrices of the copy heads show the same cutoff near 27 dimensions, since their individual spectra don't show a clear drop at 32 or 33.

I'm now comparing the top-k singular subspaces of V01 and V05, since their random components are independent and the shared directions should stand out. The principal angles between them should show how many dimensions they actually have in common.

I'm now going to compare the principal subspaces of V01 and V05, which should be independent apart from the shared token directions. I'm also considering a product-based consensus check using V01 against V05 to see which directions they really share.

