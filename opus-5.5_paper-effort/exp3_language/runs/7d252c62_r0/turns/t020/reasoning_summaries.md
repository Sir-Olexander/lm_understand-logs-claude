I'm noticing that many neurons have near-duplicate partners, with cosine similarity around 0.99 at the 90th percentile. That suggests they might form clusters of token-specific detectors. I'll try grouping neurons whose mutual cosine exceeds 0.9, count the resulting clusters, and then look at how the cluster centers relate to each other.

I'm thinking that if about 27 or 28 cluster centers come out roughly independent, they could be token detectors. But neurons might instead be bigram detectors, where this matrix only shows the current-token part and the previous-token part sits elsewhere. I'll run agglomerative clustering on the top neurons by T-norm at a 0.85 cosine threshold.

