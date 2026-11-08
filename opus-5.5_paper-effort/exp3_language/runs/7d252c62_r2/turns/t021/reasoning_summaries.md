I'm checking which heads show induction patterns, and the diagonal is strongly positive for most letter tokens but drops for a handful of positions. The value vectors in L0H5 also seem to copy only weakly from some of those same positions.

I'm checking token 22's induction diagonal in L2H1, which is moderate at 1.1, and then asking which tokens 14, 15, and 21 might be, since their induction is low. Their high kurtosis and few neurons suggest rare letters, which would make sense if rare tokens rarely need induction.

I'm weighing whether token 22 is the newline or the space: it has low kurtosis and many strongly involved neurons, which fits a very frequent token rather than a rare letter. Token 2 might be the newline instead, and I'm trying to figure out which of the two specials is the space from how L0H4's values read both equally.

I'm now looking at L1 heads, which seem to avoid keys at tokens 2 and 4 and favor token 22 in several cases. That pushes me toward thinking of token 4 as another special token, and I'm wondering whether token 2 is the space and token 22 the newline instead.

I'm checking how L1's heads treat tokens 2, 4 and 22: they seem to avoid keys at 2 and 4 and favor 22, and I'm weighing whether 22 is the space or the newline. The L0 prev-token heads point the same way, with token 2 strongly positive and token 22 strongly negative in L0H5.

I'm weighing whether token 2 or token 22 is the space, since the landmark head L0H0 favors token 2 and token 4 at the same time.

I'm weighing token 2 as the main landmark, since the last-X heads favor it and the L1 heads steer away from it, which fits it being the space. That leaves token 22 as the other special, and I'm testing whether it could be the newline.

I'm now leaning toward token 22 being the newline rather than the space: its heavy involvement across MLP0 neurons points to a very frequent, structurally important token, and in English a newline wouldn't be that prominent, so the Python reading fits better. Whatever the other special token is, it doesn't look like a letter like 'e' either, since positional heads treat it so differently.

I'm noticing that the symmetry between tokens 2 and 22 across the prev-token and value readouts is too clean to be a coincidence, so I'm wondering whether the model treats them as a contrasting pair, space versus newline, rather than one being a letter.

I'm asking whether the low-kurtosis components 10 and 14 are really separate special tokens or just leftover mixtures in the ICA, since low-kurtosis directions can be poorly identified.

I'm wondering whether ICA's low-kurtosis components 2 and 22 are just a rotated pair spanning the two special tokens, so the true token directions might be sums or differences of them. If so, the difference alone could isolate one token, which would explain the sign-flipped readouts.

I'm wondering if the 2-D special subspace might instead hold one real token plus a positional leftover, since dims 2, 4 and 13 carry some positional signal, which would make the "28 tokens" count look off.

I'm now wondering whether comps 2 and 22 are really positional directions leaking into the token subspace, since L0H4's values and L0H0's values read them most heavily. If so, the count of 28 tokens would be too high, and the true token set might be smaller, around 26 letters plus the specials.

I'm now checking whether dimensions 2 and 22 are really token directions or positional leftovers, since their P-fraction and the strong V-readout from L0H4 point that way. I'd test their overlap with the positional subspace (using a larger P, say 60 dims) and see whether the copy heads ignore them, which would lower the token count from 28.

I'm checking whether comps 2 and 22 are positional leftovers, so I'm computing, for each component, how strongly the query and key functionals of the L0 heads read it.

I'm checking whether components 2 and 22 behave like positional directions, by comparing how strongly the positional heads' query and key readouts and the copy heads' value readouts respond to each one.

