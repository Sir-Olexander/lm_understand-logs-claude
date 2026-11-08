I'm reading the cosine similarities for the top singular directions of the two sets of vectors. At k=30, they fall off sharply after about the 23rd or 24th direction, and at k=40 the drop is in the same place. So the shared token subspace between layer-0 V (heads 0 and 3) and MLP0's input looks to be about 23 dimensions.

The estimate is holding at about 24 shared dimensions, and the random-baseline tail fits that count. Taking the vocabulary as romaji gives 24 live tokens, but an English or Python vocabulary would give 28, so I need to check whether rare letters like q, x, z, and j are being pushed into the noise tail.

I'm working out the expected largest cosine between random subspaces of 40 and 70 dimensions in 379-dimensional space, which would set the noise ceiling for the k=40 tail. The gradual slide from 0.42 down to 0.34 looks more like random-level overlap than a clean cutoff, so I'm checking that against the Wachter bound.

I'm checking the random-subspace ceiling again: with p=30, q=60 in 379 dimensions, the largest chance cosine comes out near 0.64. That means the 0.46 and 0.33 values sit inside the random range, and the 0.57 to 0.93 values could be partly inflated by chance. So the knee at 23 to 24 is suggestive, but it doesn't settle the count, and I want a null comparison that swaps out S3 to see what random overlap actually looks like.

I'm going to build a random-subspace null, swapping in random directions of the same dimensions, to see what the cosine spectrum looks like by chance. I also want to compare against the P-subspace to measure how much positional contamination is inflating the overlap. Then I'll try narrowing S1 to k=26 or 28 and S3 to k=40 to pin down the token dimension more carefully.

