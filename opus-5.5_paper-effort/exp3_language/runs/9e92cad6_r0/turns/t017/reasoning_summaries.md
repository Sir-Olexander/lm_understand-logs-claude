Most of the last-layer attention heads don't write mainly into the top subspace of W_proj.3, with L3H3 the only one near a third, so their output probably goes through MLP3 instead. The output space of W_proj.3 is concentrated, with about 89% of its energy in 21 directions and 93.8% in 28. I'm also wondering whether counting tokens could work by looking at each MLP3 neuron's output vector, in case the neurons are specialized to particular tokens.

The singular values after the top 21 drop off quickly, from 9.08 down to 3.78, so there's no clean cutoff. My guess is the 21 strong directions correspond to frequent tokens, and the weaker ones to rarer tokens. Letter frequency in English could explain that: space plus 20 letters appear at roughly 1% or more, which gives 21 items, close to the 21 strong dimensions.

I'm checking whether the 21 strong dimensions line up with English letter frequencies: space plus 20 letters all sit at about 1% or more. The next few groups, a medium tier and a weak tier, would also explain the 28-dimension total.

Now I'm estimating the letter frequencies for Python code, where newlines and spaces are very common and q and z are rare. I'm putting rough percentages on the common letters and expect that only about 24 or 25 tokens clear the 1% mark, though I'm not sure of that yet.

Romaji letter frequencies would give a different count again: a, i, u, o, and the other common syllable letters probably clear 1%, with space and newline adding to the strong group. I'm not sure how many tokens that lands on yet, so I'll compare it against the English and Python estimates.

Checking whether the 4 weak dimensions are real or just part of the bulk, I'm comparing the values 6.37, 5.84, 4.63, and 4.58 against the tail that follows, 3.78 and below.

I'm checking whether the 4 weak singular values, 4.58 and above, stand apart from the smooth tail, by seeing whether those directions also show up in other output-related matrices. I'm also considering newline frequency as a way to tell an English-like corpus from Python code, since Python newlines are much more common.

Now I'm wondering whether the bulk of the singular values past index 28 is just Marchenko–Pastur noise, so I'd estimate its edge from the bulk's Frobenius norm and see whether the 4 weak dimensions stand above it.

Next I'm checking the Marchenko–Pastur edge for W_proj.3 to see whether the weaker singular values rise above the noise floor. After that I'll look at the top spectrum of W_proj.2 and how it overlaps with the up-projection.

