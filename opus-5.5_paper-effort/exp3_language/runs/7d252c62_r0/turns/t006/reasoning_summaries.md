Both heads' value matrices show roughly 22 to 23 meaningful singular values tapering off gradually, which fits copying token identity across the ~27 tokens, some of them rare. Their key matrices have strong leading components while the query matrices are more spread out, with about 20 large values in L0H5's query weights.

Next I'm weighing whether the previous-token head is a good candidate, starting from its query side.

I'm now asking whether these heads separate token and positional information: I'd compare the subspaces that the value, key, and query weights use, to see whether the query side reads position (as a previous-token head would) or token identity.

