I'm looking at heads 1, 2, 4 and 5 in layer 0, which all write into the same 2-dimensional output subspace. Heads 1, 2 and 4 seem to share query and key directions, while head 5 is close to uniform, so they may be jointly estimating something like distance since the last newline at different recency scales.

I'm reading through the heads in layer 0 and checking whether their shared output subspace is measuring how far along the current line the model is, perhaps by tracking the most recent newline at several decay scales. The heads that attend to newline tokens with recency bias would give the start-of-line position, which could then be subtracted from the current position to get the line offset.

I'm working through whether the QK bias terms matter here. I had left out ln1_b, and since the query includes W_q applied to the bias, that could add a key-only preference across the heads. The bias is small next to the weight-scaled components, but it's spread over many dimensions, so it might still shift the scores.

I'm considering uniform attention as a way to get a position signal: if one value is nonzero only at the first position, its average over the context falls off as 1/(i+1). I'd still have to check whether a token-based feature could produce the same pattern instead.

Now I'm weighing whether uniform attention with a value that's nonzero only at the first position would give a clean 1/(i+1) position signal, versus a token-based average like the fraction of newlines or spaces in the window. I'm leaning toward checking the dimension structure of the token subspace next, so I can see which of those two readings the 2-dim feature actually fits.

I'm checking the OV spectrum of L0H0 and L0H3 past the top few singular values, to see whether the effective rank is near the number of tokens in use.

I'm going to print the full OV singular value spectra for L0H0 and L0H3, plus the stacked restricted W_v.0, to see where the effective rank drops off.

