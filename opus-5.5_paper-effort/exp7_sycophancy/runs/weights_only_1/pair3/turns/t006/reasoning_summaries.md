The changes look almost identical across columns, with the top ones only about 1.5 times larger, so each unit seems to move by a similar amount. That pattern points to Adam-style elementwise updates, where every entry shifts by roughly the same small size.

I'm checking whether the per-entry differences in a matrix sit near a constant magnitude, which would fit sign-like Adam updates. I'm also weighing how the magnitudes relate to the weights, and noting that the top singular components look like real structure rather than noise.

I'm now looking at the top singular vectors of the update matrix, asking whether the directions they point to in the residual stream are the ones being read or written. I suspect the leading structure reflects consistent per-column signs on the outlier input dimensions rather than true signal.

Now I'm checking whether the top singular vectors of the update for the key projection pick out a few input dimensions, which would suggest bias-like structure rather than signal. I'll compute the top residual-side singular vectors for each matrix and compare which dimensions dominate across layers.

