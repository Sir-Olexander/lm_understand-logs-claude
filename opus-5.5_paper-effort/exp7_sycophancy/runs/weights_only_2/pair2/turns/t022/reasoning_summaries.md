I'm checking the first joint direction: it sits at 0.162 in early reads, about 13 times the null, and 0.738 in late writes, about 58 times. The second joint direction shows a similar early-read and late-write pattern, so I'm treating these as candidate answer-token directions and taking that as support for e1.

I'm now treating x1 and x2 together as a 2D answer subspace, so I'll redo the copy tests across all layers with stricter nulls. I also want to see which early-layer matrices (q, k, v, gate in layers 0 to 2) read x1 most strongly, since that should show where the token appears in the sequence. After that, I'll project x1 and x2 onto each matrix's top-k read subspace and run OV copy tests on them.

