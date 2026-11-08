Claude's weights have moved far from initialization, and the first layer's second head shows a very large query-key product with many large singular values. That points to a sharply peaked attention pattern, probably a previous-token head, since those in early layers tend to have large query and key norms in their positional subspace.

The large-singular-value pattern in that second head fits a sharp positional attention pattern, and the rank suggests many Fourier-like components rather than a token-based newline head. Next I want the full spectrum of the head's query-key product to see how many singular values are large. I also want the count of large singular values in the final projection, to check whether it matches the vocabulary subspace.

