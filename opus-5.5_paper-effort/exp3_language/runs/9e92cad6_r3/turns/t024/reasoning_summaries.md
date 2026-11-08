The head's output projection puts about 25 of its values at 2.03 or above, then they fall off sharply, so it seems to write into roughly 25 dimensions of the top-28 subspace. With k=32 the picture is the same: 25 strong values, then a steep drop. That suggests the copy head is spreading its copying across about 25 directions in logit space.

The copy head's dimensionality hinges on how many tokens it has to copy. English has 26 letters plus space and possibly newline, so roughly 27 to 28 directions. Romaji has 22 letters plus space and newline, about 24, which puts the observed 25 in between. The softmax-invariant direction likely means the effective logit space is about 27 dimensions, so the count is close to the real token set size.

Now I'm checking whether the 25 strong dimensions could match the logit space, given that the count of distinct tokens the head copies depends on which letters actually get copied strongly. English gives about 26 letters plus space and newline, which would suggest 27 or 28, while romaji gives roughly 24. The observed 25 sits between those, so I need a cleaner way to find the logit subspace, perhaps by looking at directions shared by the final-layer writers, though the MLP's contribution makes that less pure.

I'm checking whether the copy head's OV input reads the raw token embedding, since its strong rank of about 25 would then match the number of tokens it copies. I'm now going to look at the full L3H3 OV spectrum to see how the strong values fall off.

Now I'm checking the rest of the L3H3 spectrum for a cliff, and it looks smooth without one, since the head seems to write non-logit directions too. Romaji gives about 21 to 22 letters, so the count depends on whether space and newline get copied at all.

I'm checking whether the 25 strong values hold up when I look past the 28-dimensional cut, since at k=24 the 25th value came from dimensions 25 through 28. I'll rerun the projection with larger k (40, 60, 100) to see if the cliff at 25 persists or moves.

