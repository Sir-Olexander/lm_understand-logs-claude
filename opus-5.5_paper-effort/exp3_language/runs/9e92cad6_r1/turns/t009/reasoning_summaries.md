The L0H3 singular values fall steadily from about 1.13 to around 0.53 across the first 18 dimensions, then taper further without any obvious break. Checking the raw V values for the same head, they also decline gradually rather than showing a sharp drop. The strongest dimensions seem to be 18 through 20, but there's no clean gap separating them from the rest.

Next I'm weighing other ways to tell the corpora apart, since vocabulary identity is hard to read from the weights. Python code leans on newlines and copying indentation from the prior line, which is hard to spot in the weights, while it also repeats identifiers heavily, so induction heads should be strong there. Romaji text is highly regular at the character level, with vowels reliably following consonants, which points to the previous-token head doing most of the work.

I'm reading the W_proj.3 singular values as a rough count of output dimensions: about 21 strong ones, a few medium, and a tail of smaller ones. With roughly 24 characters in use (letters, space, newline), I'd expect around 23 active directions, and the rare letters might each need their own suppressing direction.

Now I'm looking at the W_proj.3 singular values: roughly 21 strong directions, then a few medium ones at 9, 8, and 6, with a small tail. Counting the output side this way, the romaji corpus would need about 23 to 24 active dimensions, while English or Python might need closer to 27.

I'm looking at L3H3 and L3H4 next. L3H4 has large output-side singular values but weak attention scores, so it probably spreads attention broadly and copies something like a bag of recent characters into the output. L3H3 has symmetric, positive matching scores and a large value path, so it may be a matching head, possibly acting as an induction-style head at layer 3.

I'm checking whether the query and key subspaces of L3H3 line up, since an induction-style match would need the query to carry the current token and the key to carry the previous-token information written by L0H3.

I'm checking whether the query and key subspaces of L3H3 overlap with a positive alignment, since that would mean the head attends to positions sharing the same feature as the current one. The real positive eigenvalues would point to that kind of matching, though it could be the same token, the same line position, or something broader like shared context.

I'm weighing whether the L3H3 query and key subspaces could share a feature like the previous token or a context representation, since an induction-style shift would need them to line up in a way that these weights may not support.

I'm going to compare which later heads read the previous-token information from L0H3, after dropping the outlier coordinates, to see whether any of them look like induction heads. I'll also check the L1 query side, where previous-token info seems to shape how it attends.

Now I'm planning to estimate how much of the T and P subspaces each later head reads, comparing that against random chance, so I can see which heads might be induction-like.

I'm going to strip out the five outlier residual coordinates (21, 39, 58, 99, 184) before measuring how much each later head reads from the token and position subspaces, so the comparison isn't dominated by those dimensions.

