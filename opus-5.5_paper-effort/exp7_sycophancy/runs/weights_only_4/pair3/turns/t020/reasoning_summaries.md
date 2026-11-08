I'm picking out L1h0, L1h3, and L1h1 as likely previous-token heads, since their attention peaks at offset 1 and drops off sharply by offsets 3 to 5, with L0h3 as another candidate. The self-versus-previous comparison is close for L1h0 and L1h3 (3.6 vs 4.0, and 4.4 vs 4.3), which suggests they also attend to themselves.

Now I'm setting up an induction test: for the later layers, I'm building a score matrix that compares query vectors at each token against key vectors that carry the previous-token information, using only the low-frequency dimensions so the result doesn't depend on distance.

I'm now setting up the induction score for the later heads: combining the query at each token with the summed previous-token output from the early heads, and checking how the diagonal compares to off-diagonal entries, normalized by standard deviation.

I'm running the same comparison across all the token sets, digits and names or words, using each model's own weights. If the first model shows stronger digit induction, the gap should appear in certain later heads, especially around layers 10 to 12 and 17, where the query and key differences are largest.

