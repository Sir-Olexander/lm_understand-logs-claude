Fin looks like the main rotation plane of L0H3, turned by about 44°, and L0H0's attention is 69% concentrated on Fin×Fin, with an asymmetric, mostly positive pattern. That makes Fin read as a strongly positional plane, probably one half of a sinusoidal (cos, sin) position encoding. Next I'm checking how L0H3 rotates it.

I'm working out what a 44° rotation would imply for the frequency and offset of this plane, and I'm checking whether averaging it over the prefix would give an early-position sink.

I'm thinking about whether a basis-invariant test could work on the token side. A bigram-style map through L0H3's OV and the later MLP doesn't seem to answer it directly. The induction head L3H3's token-space matching matrix looks close to a positive identity, which fits. I'm now considering eigenvalue spectra of a token-to-token map, since those stay the same under a change of basis.

I'm weighing whether the signature of a symmetric bilinear form, the count of its positive and negative eigenvalues, could serve as a basis-invariant test, since Sylvester's law of inertia keeps it fixed under change of basis. For same-token matching in L1, that count of negative eigenvalues would track roughly the number of distinct tokens.

I'm noting that Sylvester's law makes the signature of the restricted symmetric form invariant, so a same-token matching form would give a negative count tracking the number of tokens it actually covers. If the head suppresses every token, the count would match the full vocabulary, but if it only covers a subset like vowels, the count would be smaller. I'm going to check the eigenvalues of the symmetric part of the restricted form at several approximate token-span sizes to see how many are strongly negative.

