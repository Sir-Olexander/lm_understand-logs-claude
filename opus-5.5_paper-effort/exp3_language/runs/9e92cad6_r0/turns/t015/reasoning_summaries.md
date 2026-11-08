I'm noting the matching matrix has about 12 substantially positive eigenvalues, falling from 0.43 to 0.135, so the L0H3 OV product is effectively rank around 16. For L3H4 OV, the first 8 singular values are large, from 21 down to 11.5, then they decay gradually, which points to a broad head with large-rank output.

I'm weighing whether the previous-token signal in L0H3 is strong enough to separate corpora, since romaji's reduced alphabet might limit that. I also want to check whether the L1 clone heads, which look nearly identical in their QK and OV statistics, are really one function split across six heads, so I'm going to compare their QK matrices directly.

I'm going to check whether the six L1 heads share the same attention pattern, which would make them act like one high-bandwidth head, so I'm comparing their QK and OV matrices pairwise by cosine similarity.

