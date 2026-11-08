Looking at the attention head L0H3, its eigenvalues come in conjugate pairs at angles around 35 to 60 degrees, which points to a relative-positional rotation. I'm thinking about how each frequency component would give a different phase, since a previous-token head should shift the phase by its frequency. Many of these pairs sit at similar angles, and I'm not sure yet what that means.

Working through the algebra: if the head's Wq^T Wk acts as a rotation by θ in each frequency plane, the score peaks at offset -θ/ω, so the eigenvalue angle would fix a single positional offset. But the similar angles across different frequencies don't fit one fixed offset, which points toward an "attend to recent positions" combination instead. Either way, L0H3 looks strongly positional, since its complex eigenvalues dominate its spectrum.

Now I'm working through the rank-1 cases: a head whose query is roughly constant while its key reads one direction, so the single eigenvalue depends on how those two directions overlap. Some heads may simply be tracking whether tokens resemble the residual mean or run against it, which doesn't read cleanly from the spectrum alone. Next I'll check the layer 1 and 2 heads, where many show a shared complex pair near ±70°, and then look at L3H3, which has mostly real positive eigenvalues.

Checking whether a head with real positive eigenvalues would mean query and key read the same subspace, since an induction-style head with separate subspaces would give near-zero, nilpotent eigenvalues. So L3H3 would be a same-feature matching head, though that's a guess from the spectrum.

I'm wondering whether L3H3's positive real eigenvalues mean its query and key read the same subspace, in which case it would be matching the current token to tokens with the same representation. An induction-style head would need different subspaces for query and key, which would produce near-zero eigenvalues instead. Still, a later layer could copy the current token into a shared subspace, so this remains a guess from the spectrum alone.

Now I'm planning a composition score: for each layer-0 component, I'd take its output matrix and measure how strongly a later head's query reads that same output, to find where the residual stream subspaces get used.

I'm now setting up a composition check: for each layer-0 head, I'd multiply its OV output by a later head's key-reading weights and compare the resulting norm against a random baseline, to see which subspaces actually get read downstream.

I'm going to measure how strongly each layer-0 head's output gets read by later heads' keys, queries, and values, comparing each score against the roughly 0.05 baseline I'd expect from random matrices of this size.

