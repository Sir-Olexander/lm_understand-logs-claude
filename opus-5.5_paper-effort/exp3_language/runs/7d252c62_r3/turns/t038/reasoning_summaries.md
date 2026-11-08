Looking at the singular values for L0H1: they stay roughly flat from 2.25 down to about 1.42, then gradually decline through 1.37 and 1.28 to 0.91 at dimension 28. There's a noticeable drop after that, from 0.91 to 0.74, so I'm treating the first 28 dimensions as the stronger ones, with a tail near 0.7 that settles toward 0.45. Now I'm moving on to the same kind of check for L0H5.

I'm reading the L0H5 singular values now: they hold near 2.4 to about 1.2 at index 22, then taper slowly through 1.04, 0.92, and the 0.8s, with no sharp gap, just a gradual decline from there. For L0H1 the drop after index 27 (0.91 to 0.74) is only modest, so I'm weighing whether the strong block is really about 28 dimensions or whether a 24-dimension cutoff fits better.

Now I'm checking whether the extra dimensions past index 24 are real token-subspace directions or positional noise, since the values after that point don't show a clean cutoff.

Still checking the L0H1 spectrum for where the token-carrying dimensions end. The values past index 24 are the sticking point, since there's no clean gap there. I'm weighing whether the count is closer to 24 or 28, and whether the extra dimensions reflect the character set or position.

Working through which letters a romaji corpus would actually use: the common Hepburn set gives about 22 letters, and the rare ones (l, q, v, x) would barely move the weights. Against English or Python text, where j, q, x and z still show up often enough, the count lands closer to 24 frequent tokens plus a few rare ones.

I'm weighing whether the token rank is closer to 24 or 28, and I'll need to measure the token subspace directly rather than guess from the spectrum. The learned token directions depend on the normalization, so I'm working out what that mixes.

I'm now asking whether the positional and token directions can be told apart in L0H5's learned value rows. The token content should be the useful signal, but positional directions also get gradient through the normalized hidden state, so the measurement stays noisy.

I'm going to project out the positional subspace from V5, using Ppos from the positional heads, and then look at the singular values of what's left. If a gap shows up near 24 or 28, that should tell me whether the token rank is closer to the romaji count or the English/Python count.

