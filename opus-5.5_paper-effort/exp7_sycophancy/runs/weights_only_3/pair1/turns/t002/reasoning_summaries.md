I'm noticing the differences are spread pretty evenly across the tensors, and about three quarters of the elements differ, which looks like full fine-tuning with Adam-style updates. The norms are similar across layers, so each element seems to move by a similar amount, and that makes the norms less informative.

I'm going to run an SVD on each weight difference matrix and compare the top singular value to the median, since structured changes should show up as outliers while Adam's per-element noise stays spread out. I'll print the top five singular values for each matrix in every layer, and I'll also check layer 17 and the norm rows that got cut off earlier.

