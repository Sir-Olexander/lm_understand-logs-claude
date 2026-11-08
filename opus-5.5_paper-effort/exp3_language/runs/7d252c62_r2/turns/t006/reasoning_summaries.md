The overlaps are clear now: L0H1 and L0H5's queries and keys share a common subspace, which I'll call S_P, while their values sit in a separate, nearly orthogonal subspace, S_T. So these two heads seem to attend by matching on S_P and then copy content from S_T.

Now I'm weighing whether these two heads are positional rather than content-driven, and the answer leans toward positional: their queries and keys sit on one subspace, values on another. I'm also wondering whether a head like this would be useful on shuffled text, where the previous character is just a random sample from the line, and how sharp its attention would need to be.

I'm checking whether anything downstream reads the L0H1 and L0H5 outputs, since the K-composition with L1 heads was only at baseline. I'm considering that the L2 composition is low too, and that the score may be dominated by random init in K, so I'll measure how much of each head's output subspace each reader matrix captures compared with random, starting with the layer-0 MLP and the later-layer reads.

I'm now computing, for each layer-0 head's output subspace (top 16 left singular vectors of its OV circuit), how much energy each downstream reader matrix captures, normalized against what a random subspace would give.

