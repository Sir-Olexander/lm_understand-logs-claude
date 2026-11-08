I'm noting that I mis-printed the union's singular values, since the variable I meant was U rather than '_'. Next, the top subspace P for L0H3 overlaps strongly with L0H0's QK and is read by queries from H1, H2, H4, and H5, so it looks like a major layer-0 input direction, possibly the positional one.

I'm checking whether P tracks position or token identity, and the query side hints at position, perhaps recency. I'm weighing two hypotheses: if P is positional, L0H3 is a relative-position head, and if it's token, L0H3 is a token-to-token attention head. To tell them apart I'm comparing P's dimension against the number of significant singular values in L0H3's QK and OV.

I'm going to compare the overlap of L0H3's OV input subspace (top 18 directions) with P, and check L0H5's uniform-attention OV input against P too. If L0H5's OV input is orthogonal to P, that would support P being the positional subspace.

