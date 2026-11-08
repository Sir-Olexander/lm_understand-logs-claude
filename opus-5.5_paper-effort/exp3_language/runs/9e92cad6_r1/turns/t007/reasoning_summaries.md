I've spotted outlier dimensions 21, 39, 58, 99, and 184, which line up with the coordinates. Heads L0H1, H2, H4, and H5 write into them, and the layer 1 and 2 key directions mostly read dimensions 21 and 99. That suggests these four heads form a shared circuit that writes a 2D signal into those coordinates, which the later keys pick up on. I think this 2D signal is probably

I'm checking whether L0H5 is a uniform-attention head whose value reads a token feature, which would make its output something like a running fraction or a coarse absolute-position signal. Heads L0H1, H2, and H4 look different, with a single strongly negative QK eigenvalue, so their attention depends on one key feature.

So the shared 2-D key feature in those outlier coordinates would give every head the same key-side bias, which fits an attention sink at the first position. I'm wondering whether the massive activations come from MLP0 at position 0, so that all keys read them and the early token soaks up attention.

I'm going to drop the outlier coordinates {21, 39, 58, 99, 184} (and possibly 383, 180) and redo the composition scores, since the 2-D circuit looks like a generic sink mechanism that doesn't tell the heads apart. For the induction question, I'm now comparing K-composition of L1 heads with the L0H3 and L0H0 outputs, where L0H3 is looking strongest, including on V and Q.

I'm reconsidering the composition analysis now: the outlier coordinates are probably just a generic sink mechanism, so I'll compare the heads on their top singular subspaces instead of raw norms, which avoids the dilution from random-initialization noise.

Next I want to see which layer-1 and layer-2 heads read L0H3's output through their keys, since that is where an induction pattern would show. The Q and V composition scores with L0H3 look high, so I'm checking those against the K scores too.

Comparing L0H3's output against the later heads' keys, queries and values: the composition scores for K run around 6-7.7 against a baseline near 5, while Q and V sit higher. That pattern points to previous-token information being read mainly through queries and values, which is not what an induction circuit would need.

I'm working out whether L0H3 is a previous-token head or a local-window head, since the rotation angles across the frequency pairs look like a relative-offset pattern rather than a pure copy-the-last-token one. The thought is that a local window could instead track whether a newline just occurred, which would give position-within-line information, and shuffled indentation might complicate that picture.

I'm still weighing whether L0H3 is a true previous-token head, since its rotation angles cluster around 45° rather than spanning the full range I'd expect. Without the positional embeddings I can't pin down the geometry, and the strong rotations might just reflect whichever frequency components the head happens to use.

I'm weighing whether L0H3 is really a previous-token head: in as-is text, trigram statistics make a strong previous-token head almost certain, while in shuffled text the useful signal is just the bag of characters in a line, so the layer-0 heads would look more like broad averaging or line-local heads. L0H3's high-rank OV copying token identity fits the previous-token reading, with L0H0 as a weaker positional companion.

I'm wondering whether the complex eigenvalues reflect a positional subspace or a token-token bilinear form, since token-token forms can be non-symmetric too.

I'm checking whether the query and key subspaces of L0H3 overlap, since a positional head should read the same positional subspace on both sides, while a token-to-token head would live in the token subspace. Then I'll compare that overlap against the input subspace of the head.

I'm checking whether L0H3's value input lives in a different subspace from its query and key inputs. If the positional part is read by QK and the token part by V, that would fit a previous-token head. I'll test this with principal-angle overlaps between the top singular subspaces of each layer-0 head's Q, K, and V.

