I'm checking how the output subspaces of L0H4 and L0H0 overlap, since both are 2D and read the same positional plane. That makes them look like two versions of an averaged positional feature, with L0H0 weighting by token X. I'm also wondering how the L1 key preference, which leans on this same plane, ties into the overlap.

I'm working out whether L0H4's output plane overlaps with the positional subspace it reads from, since the numbers for L1H1 suggest it does. If so, L0H4 would be writing back into the same positional representation it reads, perhaps centering or amplifying it. That would also mean L1 keys lean on this plane with a query-independent preference, giving those heads a recency bias.

Now I'm weighing whether Python or English is the better test case, comparing how far back an induction head can reach: in Python the 128-character window covers roughly three or four lines, while in English it stays within a paragraph. I'm also considering newline prediction, since Python lines tend to end in identifiers, so predicting the newline depends on line context. A more direct check might be how many heads attend to specific tokens, with the space separator in English and the newline and spacing structure in Python.

I'm picturing whether copy heads in Python would be stronger than in English, since repeated patterns like "self.x = x" recur often. The L3H5 head has a QK matrix with all positive eigenvalues, which would make it a self-similarity matcher that attends to positions whose residuals resemble the query's. I also wonder if a head might track the position right after the previous newline, which would suit line-aligned copying in code.

I'm checking whether L0H3's positional block, which is negative and favors far-away keys, combines with its token-key preference to make it attend to earliest positions. L0H0 looks similar, with a negative positional term and a rank-1 token preference. I'm wondering whether these token-keyed heads are really keying on newlines, with some odd positional preference layered on top.

I'm weighing whether indentation-driven bigrams could show up in the MLP0 features, since it combines the current token with the previous one. In Python, repeated spaces would form that kind of bigram, while English has no equivalent. I'm also considering blank lines between functions, which would show up as consecutive newlines in the corpus.

I'm wondering whether the newline is simply frequent in Python, and whether L0H0's key token direction is a common token like newline or a rare one. I'm also checking whether L0H4's uniform averaging helps mark early positions in the window, which would hold across all corpora.

I'm checking whether one of these heads might be attending to the previous newline, the way indentation-sensitive code would need, and whether a recency preference paired with a newline key would show that. I'm also weighing whether the MLP0 weights along the token direction would tell me if that token is common in the corpus.

I'm now trying to see whether a rank-1 pattern in the token-token block, such as query newline or space attending to key newline, would show up as a dominant singular value. The L1 symmetric eigenvalues look spread out with no single dominant one, and L2H2's block is large but not rank-1. I'm also considering whether the key-previous-token by query-token blocks might show that pattern instead.

Now I'm wondering whether a sharp offset-1 attention pattern would need a positional QK rank of about 20 to 40 in any corpus, and whether a general layout of prev-token heads in L0, local heads in L1, and induction heads in L2 and L3 holds for both English and Python. I'm also considering whether newline prediction in Python, where lines average around 35 characters, depends heavily on line content in a way English text never requires.

I'm trying to recover the token embeddings by treating them as discrete points, since the composed token-to-token maps from the prev-token and induction OVs should look nearly identity-like, though simultaneous diagonalization hasn't worked due to noise. A sparse coding or ICA angle might pull the discrete token directions out of the T-subspace basis.

I'm now testing whether the MLP0 neurons cluster into token-specific directions: projecting their weights onto the 27-dimensional token subspace, normalizing, and checking whether pairwise cosines show tight groups, using k-means with 28 clusters to compare against a random baseline.

