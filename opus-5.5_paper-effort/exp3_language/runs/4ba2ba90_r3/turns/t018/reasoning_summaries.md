I'm running ICA on MLP0's T-projection, and the sources are very sparse, with kurtosis from 35 to 180, so each component seems to feed only a handful of neurons, likely token-specific ones. At 27 components, 25 of them show kurtosis above 36, while at 30 the sparse set thins out after about 27. That points to roughly 26 or 27 sparse components, which lines up with the 28 tokens I'd expect to see.

The sources' fit points to roughly 27-28 token directions, and the 24-token romaji idea looks less likely since 27 comps stay high. Still, the sparse directions might be tracking something other than tokens, like previous-token combinations.

I'm checking whether the ICA directions match what another reading matrix would produce, since shared directions would point to real token features rather than mixed prev-token signals. I'm also considering counting how many neurons load on each component, and whether rare letters like q, z, x, or j get dedicated neurons, which would show up as small sparse components.

The sparse-source evidence points to 27-28 token-like directions, which leans toward English or Python rather than romaji. To check that T isn't contaminated, I'll rerun ICA on the full 36-dimensional T-subspace and see whether extra sparse components show up as it grows.

