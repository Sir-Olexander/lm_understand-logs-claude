Checking which heads stand out: L3H3 has a strong positive trace with L0H3 and several L1/L2 heads, which fits an induction head pulling current context against previous-context features from earlier local heads. Next I'm assigning labels to the L0 heads, with L0H3 marked as previous-token.

Now I'm weighing whether the L0 and L3 heads I haven't labeled yet could instead be recency-based local heads, and checking their score ranges to see if they're really broad.

I'm settling the final answer's probabilities: English at about 0.65, Python around 0.2, and romaji around 0.15, with Python ruled down by the missing token-keyed L0 heads and the single late induction head. Ordering is as-is at 0.88 and shuffled at 0.12, and I'll write it with those numbers quoted.

