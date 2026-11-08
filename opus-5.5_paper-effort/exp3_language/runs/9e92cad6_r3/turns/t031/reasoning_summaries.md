The MLP top-k output subspaces look barely distinct from random baselines in L2 and L3, apart from the ones reading mlp1 and mlp2, so they're hard to decode further. L3H3 notably avoids reading MLP outputs, with its q/k/v below baseline for mlp2. That points to it reading raw embedding and L0 attention features instead, which fits an induction head: the query reads raw tokens, the key comes from L0H3's previous-token signal, and the value reads raw input.

I'm checking whether L3H3's query and key really pick up the expected signals. The numbers from run 13 are only mildly elevated over baseline, so I want a cleaner statistic and a randomized control against random subspaces to see whether the positive-definite block is actually meaningful.

