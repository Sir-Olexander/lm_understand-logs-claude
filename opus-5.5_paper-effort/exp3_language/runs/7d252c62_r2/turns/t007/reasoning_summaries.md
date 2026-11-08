I'm checking which downstream layers read the L0H5 output: L1 values, queries and the MLPs get most of it, while L1 keys barely use it. That means there's no K-composition with a prev-token head, which classic induction would need. Instead, the L1 keys seem to read from L0H4 and L0H0 through their special dimensions.

I'm weighing whether these heads are really induction heads. A model this size would normally show clear K-composition with a prev-token head, and its absence makes me wonder if the training text was shuffled. Maybe the positional-looking heads are doing something else, like matching on token identity in the queries and keys while their values carry position.

I'm checking whether the token subspace of L0H5's value is capped near 27 dimensions by looking for a drop in its singular values there, and I'm checking the same kind of spread for the Q and K matrices.

I'm now looking at the L0H4 value output, which seems to have rank 2 or 3, with its query and key nearly flat so attention is roughly uniform. The L1 keys lean heavily on that output, so those heads may be attending by a running average of a couple of features. One idea is that the value reads the positional embedding, so the first positions stand out and become an attention sink near the start of the window.

I'm checking whether the L0H5 and L0H1 outputs feed the L1 keys the way classic induction would need. They barely do, so I'm weighing whether this is really an induction setup. The L2 and L3 key reads from L0H5 are also weak compared to the value reads, which is the opposite of what I'd expect from a clean induction circuit.

Still weighing whether the lack of K-composition from prev-token heads means these layers aren't doing induction, which would point toward shuffled training text. The L1 and L2 key reads from the L0 heads look weak, so the L0H4 sink-like behavior may be carrying most of the attention routing. That leaves open how the L0H5 and L0H1 copying would be useful in that case.

I'm trying to pin down what L0H1 and L0H5 are actually encoding in position, since a clean prev-token head would show a shift operator between query and key positional components. Without seeing the positional embeddings directly, I can only infer that from the weight structure. Both heads share overlapping query subspaces, so I'll check whether they split the positional and token roles between them.

I'm now thinking about whether the L0H1 and L0H5 heads act as offset-1 and offset-2 previous-token heads, since classic induction setups often stack those to build trigram features. The L1 value reads from L0H5 are strong, so the L1 heads may be moving "previous character" information from key positions to the query, a bit like an offset induction. If the text were shuffled, the positional heads might just be averaging over the local context instead.

I'm checking what L0H0 looks like, since its QK scores are tiny but its Q and K have a couple of strong directions, and its output lands in the same special dimensions as L0H4 and L0H3. Then I'm wondering whether L0H4's value reads a newline-like token direction, which would make these heads track line boundaries.

I'm wondering whether the newline and space tokens might each show up as their own cluster in the token subspace, since they're frequent. Another angle is to look at the layer-0 MLP input weights, which often act as token detectors, and see if projecting them onto that subspace groups them into about 28 clusters.

I'm wondering whether the corpus can be pinned down from the line structure: Python would show short indented lines with repeated identifiers, romanized Japanese would have long paragraph lines with a strong consonant-vowel rhythm, and English Wikipedia would have very long paragraph lines. Shuffling would change how those line lengths look, so that is worth checking too.

Now I'm weighing how line length might differ between the corpora. Long lines in Wikipedia and Aozora would mean a shuffled window is mostly one line, so the model would need to average over the whole context, while short Python lines would call for line-local attention. L0H4 looks like the uniform averager, though its low-rank output makes me doubt it carries a full bag of characters.

I'm checking whether L0H4's value reads the token subspace or the positional one, by comparing its overlap with L0H1 and L0H5. My plan is to find where the singular values drop for the joint token subspace of those heads' values, and the joint positional subspace of their queries and keys, then see which of the two L0H4's value leans toward. I'm also considering whether the token and positional embeddings themselves would give a cleaner split.

I'm reasoning about whether the massive-activation pattern in L0H4 could act as an attention sink: a uniform head that writes a large constant into a few residual dimensions would make early tokens stand out to later layers' keys, even though a plain average over tokens wouldn't carry much token information on its own.

I'm considering whether L0H4's output acts as an early-position signal: a running average whose magnitude shrinks as the window grows, so the L1 keys could pick out the first few positions as a sink.

I'm thinking the output of L0H4 might act as a positional signal rather than a token one: if its value reads a direction that is large only at position 0, its running average would decay like 1/(j+1), marking how far a token sits from the start of the window. That kind of sink mechanism would show up the same way in both the as-is and shuffled data, so it doesn't tell me which corpus this is.

I'm asking whether the L0 heads could support induction through the query side instead of the key side. That would have the query read the previous character while the key reads the raw token, but the value at a match only gives back the previous character, not the one that follows it, so this route doesn't quite fit an induction copy.

I'm asking whether L0H5 might attend forward or to recent positions instead of acting as a plain previous-token head, though it still reads as positional either way. Setting that aside, I'm turning to the shuffled-data case: the useful signals there would be the character bag of the current line, the bags of earlier lines, and where the current position falls relative to typical line lengths.

Checking whether the L2 and L3 heads with strongly positive QK scores might be doing something different from the earlier layers.

I'm asking whether the symmetric matching could still produce induction: if both query and key residuals encode the same current and previous token, attention would land on an earlier occurrence of the bigram, but the value at that position only sees what's causally available there, so copying the following token doesn't come for free.

I'm asking whether symmetric matching, where query and key encode the same current and previous tokens, can give induction. Attending to an earlier occurrence of the bigram doesn't help, because the value at that position can't contain the token that follows it. So positive-real QK scores would point to same-context attention rather than an induction mechanism.

I'm now estimating the effective dimension of the positional QK subspace for L0H5 and L0H1: the K singular values decay slowly, and the Q values stay flat over about twenty dimensions. That would suggest a fairly high-frequency positional code, which a recency-style head could use on shuffled lines. I'm also considering whether L0H5 and L0H1 might read content instead of position.

I'm now wondering whether L0H5 and L0H1 might instead attend to newlines and copy their positions, so each token learns where its line started. If that's the mechanism, L1 could restrict attention to the current line by matching those line-start signals, and the same L0 head would show up on both the key and query sides.

I'm now checking whether L0H4's value output reads the token subspace or the positional one, by comparing its top directions against the subspaces of L0H5 and L0H1. If the positional reading is flipped, the same test would reverse the conclusion, so I'll keep that in mind.

I'm going to check whether the singular values of the stacked L0H1 and L0H5 value outputs, and separately their stacked query and key matrices, separate cleanly into a token subspace and a positional one. I'll compare the principal-angle overlap of their top 27 dimensions, then see where L0H4's value output projects.

I'm going to project L0H4's top three right singular vectors onto the same token and positional subspaces, to see which one its value output lines up with.

