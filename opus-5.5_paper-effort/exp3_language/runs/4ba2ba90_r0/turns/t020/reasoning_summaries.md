I'm looking at the singular values of W_proj.3: 23 large ones falling off to 10.37, then a band of five medium values around 8, then a few smaller ones before the noise floor near 3.22. That puts the significant directions at roughly 28 to 30. I notice 23 plus 5 comes to 28, and I'm wondering whether those five medium values might be the unused directions.

I'm checking whether the five medium values could correspond to rarely used tokens. If the romaji vocabulary has four rare letters and the English one has a few low-frequency letters, the spectrum might be shaped by how often each token appears.

I'm wondering whether the same kind of spectrum shows up in other output-writing matrices, so I'm comparing W_o.3's left singular vectors, which have no clear cliff, against W_proj.3. The MLP3 output looks like the main logit writer, and I want to see if its top subspace matches the 28-dimensional span of the unembedding rows.

I'm thinking about whether the neuron outputs could be made sparse by finding a basis of the 28-dimensional space. If I could recover the unembedding directions that way, I'd know which tokens each neuron promotes, though I haven't settled how to do that without already having the unembedding.

I'm working through how to recover the individual token directions from the unembedding-like subspace, maybe by looking for sparsity among the neuron output vectors, though that needs labels I don't have. Counting the likely used tokens per language gives about 23 to 24 strong dimensions for romaji, and 27 to 28 for English and Python. Next I want to compute the W_proj.3 spectrum more carefully.

I'm pulling the full singular value list of W_proj.3 to see where the spectrum drops off, and I'm also considering whether the input-side matrices could give a cleaner token count.

I'm checking the tail of the W_proj.3 spectrum: the values keep falling through roughly 11 and 12, then into the 8s and 5s, with the 3.22 floor below. That gives about 30 values above noise, so two more than a clean 28-token count would predict. I'm wondering whether the extra directions come from layer norm rather than from the vocabulary, and whether rare tokens like l, q, v, and x would still get output directions at all.

I'm working out why rare romaji tokens would get weak output directions: their probabilities stay near zero, so the MLP3 gradient barely touches them. That suggests roughly 24 strong directions for romaji, though softmax shift invariance may remove one.

The count works out: romaji uses 24 tokens, and since the output probabilities sum to one, the gradient only spans 23 independent directions, which lines up with the 23 strong singular values. The five medium values could be the rare letters, which still get small gradients because they show up occasionally.

I'm running through English letter frequencies to see whether the rarer letters like q, z, x, and j would show up as a weaker gradient, the way the rare romaji tokens do.

I'm working through English letter frequencies: the common letters and space form a large group, while newline, x, j, q, and z appear far less often. That could mean the English case also gives about 23 strong directions plus a few weaker ones, the same pattern as romaji.

I'm counting the English letters now: the frequent group gives about 22 strong directions, and newline, x, j, q, and z fall into the weaker group. For romaji I'm tallying the used tokens, which comes to 24, with l, q, v, and x as the rare ones, so I expect around 23 strong directions there too.

Now I'm comparing letter frequencies in romaji and Python identifiers, since the rarer tokens in each might explain the weaker singular values.

I'm estimating how many tokens are frequent enough to produce strong singular directions in English, counting the common letters, space, and k as one group. I'm checking whether that frequency-based count matches the observed split between 23 strong and roughly 5 medium values.

I'm weighing whether the gap in W_proj.3's spectrum after the 23 strong values would separate frequent tokens from rare ones, since English has about five low-frequency tokens (newline, x, j, q, z) that might sit in a weaker group. I'm also noting that the extra above-noise directions could come from layer norm effects rather than the vocabulary, so the count stays uncertain.

I'm checking whether the spectrum could come from a different output writer, like the union of all layers' contributions to the output subspace. I'm also considering whether sparse coding of the MLP3 neurons could recover the individual token directions, which would make the relation between tokens easier to see.

I'm turning to the input side now: the L0H2 OV spectrum, where the first ten singular values run from about 0.79 down to 0.52 before dropping to zero. I'm wondering whether the input-token subspace shows the same kind of cliff, since unused romaji letters like l, q, v, and x would leave their embeddings unread, putting the input count near 23 or 24 dimensions and English nearer 27.

I'm checking the tail of the W_proj.3 spectrum where it drops from 0.393 to 0.347 to 0.304, which isn't a clear cutoff. Rather than read the spectrum further, I'm considering projecting the neuron output columns onto the top 30 left singular vectors.

I'm thinking about projecting the neuron output vectors onto the top singular directions, then looking for a sparse basis with ICA so each component lines up with a single token's unembedding. That would let me group neurons by which token they promote, though attaching labels to those groups would need more than the weights alone.

I'm looking at how the induction head L3H1's OV circuit maps input token embeddings to output logits, since it copies the attended token straight into the logits. That could link the input-side and output-side bases, which would let me pair each MLP3 neuron's current-token input with the token it promotes.

I'm considering whether the L3H1 OV spectrum can give the number of input tokens in use: copying only matters for tokens that appear, so its rank should track the used-token count. I'll print more of that spectrum to check.

