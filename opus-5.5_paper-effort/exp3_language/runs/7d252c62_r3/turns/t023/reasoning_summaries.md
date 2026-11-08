I'm noticing the top dimensions line up with the large ln1_w.0 weights, with a correlation of 0.92. That may be partly an artifact, since I computed the subspaces from ln1_w-scaled matrices, which inflates those dimensions. I should redo this with unscaled directions in the raw residual space, so I'm writing out the bilinear form on normalized x with the diag(w) scaling kept explicit.

I'm checking whether the positional directions really sit in the x-space dimensions where the LN weights are large, since the scaling by w is the actual sensitivity and not just a bias. The open question is whether the positional information is confined to a set of residual dimensions, and whether the large weights there amplify position for attention. I also want to know whether the positional embeddings are much larger in norm than the token embeddings in those dimensions.

I'm considering whether the positional embeddings could be recovered from the eigenplane structure, but the period isn't pinned to 128 since nothing forces the frequency onto a 2πk/128 grid. One idea is to test candidate frequencies by checking which choice gives the sharpest attention at a fixed offset in layer 0. I'm also wondering whether L0H1, which would be useless for previous-token lookup on shuffled text, might instead be a local-window head.

Now I'm checking how much downstream layers read the output of L0H1 compared with L0H5. The read gain is what I'm measuring, with MLP0 and MLP1 as the first readers, and L1's value projection also in the picture. So far o0.1 looks weakly read by everything, and I'm still tallying the numbers.

I'm checking how much the downstream layers actually read o0.1 compared with o0.5, since prev-token information should matter a lot in an as-is model. Measuring the absolute size of the W_fc.0 and ln2 path per dimension, rather than the enrichment ratio, should avoid the random-init inflation. I also want to see whether o0.1 overlaps with the token subspace T.

I'm weighing whether the kernel can pin down the positional offset, and I'm considering whether L0H1 and L0H5 are simply positional heads without a clear prev-token or local-window signature. Next I want to check later-layer heads for content-based attention, such as L3H5, which has a strongly symmetric, positive QK structure that suggests same-feature matching.

Now I'm checking whether L3H5's query and key read the same features, and whether its output copies token identity. If it does, that would fit an induction-style head that looks up earlier occurrences of the current token.

I'm checking whether L3H5's key side reads the same features as its query, and I'm wondering if an earlier head shifts information from the previous position into those keys. L2H2 and L2H3 look like they attend by position, since their keys read the positional plane strongly, so I'm testing whether they act as previous-token heads.

I'm looking at the QK eigenvalues for L0H4 next, where a rotation in the f-plane could mean position-based attention. If the code really rotated by about 73° per position, averaging over positions would wash it out, which conflicts with the uniform-looking pattern I'm seeing. Maybe L0H4 is less uniform than I assumed, since its QK is small in the w-scaled form, and its key preference is zero.

So the uniform attention means f(t) is just a running average of two scalar features, which changes little at large t and so can't carry a previous-token offset. I'm wondering whether those features might instead pick up something like newline tokens rather than position.

I'm considering whether the positional term could be a running average of two scalar features, with position 0 standing out since its embedding may have a large norm. If so, the average would decay roughly as 1/(t+1), giving a coarse absolute-position code, though it still doesn't settle the question.

Now I'm looking at the singular values of L0H1's OV matrix, which should show how many distinct tokens matter for this prev-token head and how strongly each contributes.

I'm looking at the singular value spectra of the OV matrices, where the drop-offs might show how many distinct tokens each head actually uses. The gap after roughly the 23rd value in L0H1 and around the 20th in L0H5 could reflect the active token set, and I'm checking whether that fits the 28-dimensional token space after removing the mean direction.

Now I'm weighing whether the token frequencies in romaji versus English could show up in the OV spectra, with a gap in the singular values around the 23rd value. I'm thinking the spectrum alone doesn't discriminate well between the two corpora, so I'd need a direct proxy for how often each token appears.

Next I'm weighing whether newlines get special treatment, since in Python the start of a line needs to look back at indentation, while English paragraphs and romaji dialogue would have far fewer line breaks. I'm considering whether L0H2 or L0H3 might attend from the current position to tokens that are newlines.

I'm now checking whether L0H2 and L0H3 key on a single token direction, which would mean they look for the most recent newline or space. I'll restrict the key side to the token subspace and check the rank of that map, then see whether the same key direction shows up across other heads.

