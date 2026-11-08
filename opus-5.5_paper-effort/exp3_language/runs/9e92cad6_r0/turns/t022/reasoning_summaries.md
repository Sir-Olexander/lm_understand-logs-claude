I'm noting that the MLP0 input is mostly orthogonal to the positional subspace, and that it carries the value inputs for L0H3 and L0H0. L0H3's query-key structure sits in the positional-by-positional block rather than the token block, so I'm splitting the space into a roughly 26-dimensional token subspace and a positional subspace, and now I'm working out how the remaining pieces fit.

I'm checking the heads that carry token information and the ones that work on position. L0H3 is attending within positional space to move token information, so it acts as a previous-token head, and L0H0 seems to do something similar with a shorter, local offset. The remaining heads read mostly positional input, and their keys don't depend strongly on token content, so they look like positional feature processors. The token subspace is around 26 dimensions, which fits a 28-token vocabulary better than a 24-token romaji one, unless some positional leakage is inflating it.

I'm checking whether the 26 token directions in Tm contain any positional leakage, since the positional subspace may be bigger than the 20 dimensions I had for P. I'll look at the singular values of L0H3's and L0H0's OV circuits restricted to Tm, to see how many of those directions the token-moving heads actually use.

I'm checking whether the rank of the OV circuit restricted to the token subspace matches the number of tokens, which would tell me how many token distinctions the previous-token head actually carries.

