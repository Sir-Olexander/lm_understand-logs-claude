I'm noticing that many neurons have T-weights pointing in nearly the same direction, with cosine above 0.9. The T-norm distribution is skewed: most neurons read T weakly, while roughly 10-15% read it strongly, which could mark token detectors. There are about 31 clusters at that threshold, though they may mix sign-flipped and multi-token combinations, so I'm considering ICA as a cleaner way to separate them.

I'm going to take the roughly 300 neurons with the largest T-norms and check whether their T-weights look sparse in a token-dual basis. I'll try FastICA with 26 components and check their kurtosis, and I'll also compare clusters using signed cosine at 0.95 to see how many distinct directions there really are.

