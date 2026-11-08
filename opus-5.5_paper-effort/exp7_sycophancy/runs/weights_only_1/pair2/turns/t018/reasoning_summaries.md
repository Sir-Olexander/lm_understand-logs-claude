The token-subspace layers look slightly negative while the complement is positive, so the strong positive trace in layer 17 must come from outside that subspace.

I'm reconsidering the approach, since the copy-evidence test is still inconclusive. The idea is to exploit the structure of the products: the second-order cross terms break the symmetry, but they are very small. I'm also considering whether the common component of the two fine-tunes could be separated from the base weights, though it looks hard to isolate.

I'm looking at which positions have A equal to B, since those carry no signal, and where they differ the change is symmetric. The idea is to estimate the common update C and compare the difference D against it, though isolating C from the base weights still looks hard.

I'm weighing whether the common update C could be recovered from the bf16 equality pattern, since where A and B match they may share a fine-tuning component. But C is buried in A+B alongside the base weights, so it doesn't look separable yet.

I'm thinking about why the designers removed the token embeddings and tokenizer: it suggests they expect a weights-only analysis to work. The difference between the models looks spread out evenly, so I'm considering whether the real signal sits in a few components buried under Adam noise, possibly isolated by looking at what's consistent across elements.

I'm checking whether the top input directions of ΔK and ΔV line up in each layer, since a shared direction would point to a guess-reading circuit. If they do, I'd then look at which heads' query changes pair with that key direction.

I'm checking which subspaces of the attention and MLP updates line up across layers, looking at whether the input directions of ΔK, ΔV, ΔQ and the gate/up changes share axes, and whether the output directions of ΔO and Δdown do too, to spot cross-component coordination.

