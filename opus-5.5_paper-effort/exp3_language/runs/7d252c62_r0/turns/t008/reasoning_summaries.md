I'm noticing that the layer-1 heads' direct token-to-token attention has a strongly negative trace, so their queries tend to avoid keys carrying the same token as the query. That's an odd pattern, and I'm not sure yet whether it's an anti-self-matching effect in the as-is text or something about predicting from a shuffled multiset. I also want to check the token subspace coming from the layer-0 OV inputs of those two heads.

The layer-1 token-token pattern is negative, so those heads seem to attend away from keys matching the current token. The layer-2 and layer-3 heads show a strong positive match between the query's token and the layer-0 H5 output, which would fit induction if H5 is a previous-token head. I still need to check whether H5 is actually behaving that way.

I'm checking the induction story from the other side now: if L2H1, L3H1, and L3H5 are copying the token at the matched position, their OV circuits should read from the token subspace. L3H5 shows a modest token-subspace read (about 0.125 against a 0.06 baseline), L3H1 is similar at 0.106, and L2H1 is zero.

Now I'm checking whether the layer-1 heads' negative token-token trace, the induction-style matching from L0H5, and the modest token-subspace reads in the layer-2 and layer-3 heads all fit together. L0H5 looks like the likely previous-token head, with L0H1 possibly a different positional offset. The L1 heads also draw heavily on the layer-0 H1 and H5 outputs, along with positional reads from L0H4.

I'm checking whether L0H1 or L0H5 is the true previous-token head, since the strong match from L0H5 might instead mean L0H1 is a two-back offset. I want to test a joint key-query match where the query's previous-token output lines up against the key's L0H1 output.

I'm planning a full table comparing each query source (token, the layer-0 H1 output, the layer-0 H5 output) against each key source across the layer-1 to layer-3 heads. If the query's H5 output, which carries the previous token, lines up positively with the key's H1 output, that would mean H1 reads from two positions back and H5 from one.

I'm now asking what could tell the Python, romaji, and English corpora apart: the number of active tokens, indentation at line starts, and how long lines run. For the token side, I'm looking at the token-related weights, including how the layer-0 MLP reads the token directly.

I'm trying to pin down the token subspace more concretely by pooling the input directions of every L0 matrix that reads tokens, and I'm considering whether the attention-output directions should be folded in too. I'll look at the full singular value spectrum of the stacked OV matrices for the two L0 heads to see where the cutoff really falls.

I'm checking the singular value spectrum of the stacked OV matrices for the two layer-0 heads to see where the knee falls, since at most 28 token vectors can be independent. A knee near 24 rather than 28 could hint at a romaji-style alphabet, though rare letters like q, z, x, and j still get enough occurrences in English to be trained.

I'm working out how many distinct token vectors the layer-0 OV maps can actually separate. Romaji text would use about 24 active characters, while English or Python would use 28, so the singular value knee in the stacked OV matrices should show which one fits. I'm also keeping in mind that LN centering alone doesn't drop the rank of the token set.

I'm trying to isolate the token subspace more cleanly, since the L0H1 singular values don't show a sharp cutoff near 24 or 28. The tail keeps going past 28, which suggests positional contamination. I'm considering whether composing with later layers would separate the token signal better, or whether pairing the H1 and H5 OV maps would help.

I'm checking whether the token subspaces of the layer-0 heads really shrink to about 20 dimensions, since they share a strong overlap in their top directions. To test that more carefully, I'm planning to compare the principal angles between the Wv row spaces of H1 and H5 over a wider range of ranks, and to pull in the value matrices of H2 and H3 as well.

I'm going to check the L1 heads' direct token-to-token QK, which is large and strongly negative, and then look at the layer-0 MLP. The W_fc.0 neurons read token, position, and attention output, so I want its singular values to see which directions carry the token signal. After that I'll build the composition table to settle the offsets, and then count the tokens.

