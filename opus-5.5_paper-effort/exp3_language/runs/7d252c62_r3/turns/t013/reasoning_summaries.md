I'm looking at the L1 heads, where the query-independent key preference sits mostly in the o0.4 subspace, a 2D part carrying about 0.95 of the norm. The preference's magnitude is around 2 to 3 score units after dividing by 8, which seems plausible for a few units inside that subspace, so these heads appear to prefer certain averaged-position keys regardless of the query.

I'm working out what this query-independent key bias actually does: since it doesn't depend on the query, it acts as a fixed function of key position, possibly favoring recent tokens or an early position as a default sink. The query-dependent parts are spread out without a strong subspace pattern, so I need to look there next.

I'm checking whether the later layers read prev-token information through their keys. Looking at the K-composition scores for L2 and L3 against the L0 prev-token heads, they come out near baseline, which seems to argue against a plain induction mechanism, unless MLP0 is reprocessing that information first.

I'm checking whether L0H1 and L0H5 might be local averaging heads rather than previous-token heads, and looking at where their outputs go, mainly through V-composition into the layer-1 heads. Then I want to see which downstream weights, including MLP0's input, read the prev-token subspaces, since MLP0 could be where bigram features get built.

I'm now asking whether MLP0 combines the current token with the previous one, since bigram-style features would be built there. To separate previous-token from local-averaging behavior, I need to find positional structure, so I'm looking at whether the positional subspace traces a curve that the L0H1 and L0H5 attention patterns depend on.

I'm testing whether the query-independent bias could be a plain previous-token mechanism, but the smoothness assumption on positional embeddings makes this underdetermined. I'm considering whether the symmetric part of the positional bilinear form, which dominates for nearby positions, would make the heads avoid self-similar positions and attend farther away instead.

I'm comparing the symmetric and antisymmetric parts of the positional bilinear form, since a previous-token head should behave like a rotation in each frequency plane. The sign of that rotation would tell me whether the head looks back or forward, but I can't pin that down without knowing the positional vectors. So I'm checking whether L0H1 and L0H5 share the same positional planes as a way to test their orientation.

I'm weighing whether a shuffled-text control would help here. Without induction or copy heads doing useful work, I'd expect the later layers' QK to lean on broader positional or line-structure features. The strong symmetric positive QK in L3H5 would then point toward attending to similar tokens, which is where I'd look to separate the as-is and shuffled cases.

I'm weighing whether L3H5's symmetric positive QK could be a "same-character" or duplicate-token head, which would attend to earlier occurrences of the current token. The causal constraint makes a true induction copy hard to see here, since the next token isn't yet visible at the key position, so the symmetric form seems to point elsewhere.

I'm checking whether MLP0 reads the prev-token outputs from attention 0, since combining the current token with a copied neighbor would produce bigram-like features. The trouble is that previous-token and local-window behavior look alike from the weights alone, so I'd need to reconstruct the positional geometry to tell them apart.

I'm testing whether a previous-token head would look like a rotation matrix acting on sinusoidal positional embeddings, since then the head's query-key form would peak at the immediately preceding position.

I'm checking whether M1 and M5 share eigenvectors within the positional subspace, since if both are roughly powers of the same shift operator, their eigenangles should scale with the offset ratio. To test that, I'd restrict both matrices to the 24-dimensional positional subspace and compare M5's action in M1's eigenbasis.

I'm going to check whether M1 and M5 share eigenvectors within the positional subspace, by computing the eigenstructure of each restricted to Ppos and seeing how the other acts in that basis.

