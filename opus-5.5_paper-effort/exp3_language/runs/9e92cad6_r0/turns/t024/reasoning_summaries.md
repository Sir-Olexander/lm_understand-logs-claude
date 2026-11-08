I'm checking the first plane, where L0H3 and L0H0 sit near -40°, and the later layers' heads land around -75° to -97°, roughly twice L0H3's offset. L3H3 is close to zero, so it may be sitting at the same position.

If the offset is Δ=1 for L0H3, then L0H0 looks similar in this plane, and the heads in layers 1 and 2 sit at roughly twice that angle, which would put them around two positions back. The dominant plane sets the frequency at about 44° per position, roughly an eight-position period. L0H0 also has a strong rank-one component that may be a bias-like preference rather than rotation, which would contradict its other planes, so I need to check that more carefully.

I'm checking whether the later heads' positional offsets are roughly twice L0H3's, which would make them attend about two positions back, and working out how a dominant frequency near 44° per position fits that.

Looking at L0H0 in the same main plane, its angle of about -38° matches L0H3's, which suggests a previous-token-like offset too. Its other planes don't show the same rotation, though, so the strong rank-one component may be a recency bias instead, and I'll treat L0H0 as a softer, local head.

I'm labeling the remaining heads by role: H5 as broad, and H1, H2, and H4 as broad with weak recency. The layer 1 heads look local, around two positions back, and the layer 2 heads may be local too, though their main positional signal seems to come from the later features.

I'm now checking the layer 3 heads, H0, H1, H2, and H5, along with L2H3, since their QK blocks all sit near baseline and they seem to read from later-layer features rather than the positional planes.

Now I'm checking whether the layer 3 heads, starting with L3H0, L3H5, L3H1 and L3H2, lean toward positive content matching, since their positive QK values outweigh the negative ones, which would make them look more like copy-style heads. L3H0's K-composition with L0H3 comes out negative, which I'll keep in mind. Next I want to see how these layer 3 heads compose with the layer 1 and layer 2 outputs, which carry the previous-token information.

Next I'm checking whether the layer 3 heads read the layer 1 and layer 2 outputs, meaning whether their keys match the two-tokens-back information those earlier heads carry. To test that, I'll compute the trace of the composed QK and OV matrices, normalized by the Frobenius norm, and compare each against the baseline.

