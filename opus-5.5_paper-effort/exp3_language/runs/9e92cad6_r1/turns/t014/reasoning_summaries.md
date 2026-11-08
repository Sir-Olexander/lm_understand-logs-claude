I'm looking at how the L3 heads' OV outputs line up against the top MLP3 output directions, and L3H3 is the one that overlaps most. The L3H4 values jump at k=26 through 28, which makes me think the W_proj.3 singular directions there are shared with that head's output. I'm starting to wonder whether attention head outputs also write to non-logit directions

I'm working out whether the L3 MLP's top output directions can serve as a stand-in for the logit space. If the learned columns of W_proj.3 mostly fall in an m-dimensional subspace, the singular value spectrum should show that, so I'm planning to check the spectrum to estimate how many logit directions there are.

I'm comparing the singular values of W_proj.3 to see whether there's a clear break separating the top block of directions from the rest. The drop after the first 21 values is small, so it doesn't cleanly mark off a logit subspace. I'm also weighing whether the rare directions just have smaller singular values because fewer neurons use them.

Now I'm weighing how the romaji letter set compares to English and Python, since the rare tokens are about the same in each case. That means the top cluster size alone won't tell me the logit count. If romaji has 24 live tokens, the meaningful spectrum should stop around there, but the values past 28 are still above what I'd expect from random directions.

I'm checking whether the top ~28 singular directions of W_proj.3 are cleanly separated from the rest, by looking at how concentrated each direction is across neurons. The thought is that rare-token logit directions would be written by only a few sparse neurons, so they should stand apart from the common ones.

I'm weighing whether the induction head's matching eigenvalues can reveal how many tokens actually appear, since tokens that never occur give the head nothing to match. The top eigenvalue sits near 0.43, with a few more pairs stepping down toward 0.24.

I'm thinking about whether the spread of the lower-order L0 heads' attention, which depends on token-type keys, could signal something about position, and whether a query-side scaling factor makes the attention sharper as the position grows. That would be more informative than counting dimensions in the matching spectrum, which only needs a few dimensions to tell tokens apart anyway.

I'm working out how a recency-sensitive attention pattern could come from a rank-1 query-key product, where the positional term grows with token index. If the value side copies positional information, the output could encode how far back the last matching token was, which would land in the outlier dimensions.

I'm looking at whether the outlier dimensions in the key vectors could encode the position of the last matching token, so that positions within the same segment share a key value. If so, a query could score segment positions against each other, which would give the layer-1 and layer-2 heads a way to attend within a word or line, or to the previous segment.

I'm comparing the L3 induction head against the other heads, since in a four-layer model induction usually shows up in layers one or two, and here only L3H3 seems to carry it. I'm also weighing whether the corpus type, repetitive code versus natural words, would change how strong the n-gram matching should be.

I'm weighing whether the L1 heads' strong negative QK pattern, where a query token avoids keys of the same token, could reflect a rule like attending to the opposite class, such as vowels against consonants, and whether that would hold across languages.

I'm treating the negative "same token" pattern in the L1 heads as a possible vowel/consonant split, where each query attends to the opposite class. If that's right, a rank-1 negative direction along that axis should dominate the eigenvalues for those heads, so I'm going to print the top eigenvalues of the same-token matrix for each L1 head and see whether one stands out.

Checking whether the L1 "same token" patterns share a dominant axis across heads, like space versus letter, would be a good test, so I'll compare their leading directions.

