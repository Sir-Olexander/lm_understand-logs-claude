I'm comparing the attention read norms across layer 0 and layer 1, with newline and space looking closest in layer 0 and newline standing out in W_v.1.

I'm weighing whether newline's heavy read in layers 0-1 points to Python over English Wikipedia, since its line structure and indentation are central there. Romaji text would also have newlines followed by spaces from indentation, so I'm keeping that option low at about 0.12. I'm also looking at a third token, comp 4, which shows strong reads in several weight matrices.

I'm checking whether comp 4 is a special direction in ICA space that also accounts for the space token, and whether the as-is vs. shuffled call holds up. The induction heads (L2H1, L3H1, L3H5) strongly favor as-is, around 0.93 probability, but I want to rule out that the positive k05 signal could also appear under shuffled training through a local-window head.

I'm now checking whether the head at layer 2 that matches on the previous-position token shows a negative same-token score alongside its strong positive k05 signal, which would mark it as tracking the prior token rather than the current one. Shuffled text would give no reason for that pattern, so this supports the as-is call. Next I'd like per-head labels for layers 1 through 3, sorting each by how much of its QK behavior is positional versus content-driven.

I'm now going through the attention heads in layers 1 through 3, starting with a breakdown of how much each head's query-key scores come from positional versus content directions. For layer 1, the heads look like they track relative position through the slowly varying landmark features, which points toward broad attention rather than strictly local.

I'm going through the layer 1 heads now, leaning toward calling them local since they seem to attend within the current word or recent segment, relative to the landmark features. Head L1H0 looks weaker and more uniform, with a smaller QK spread, so I'd tentatively label it broad. The others in that layer look stronger.

I'm going through the remaining layer 2 and layer 3 heads, leaning toward calling L2H3 and L2H4 local since their key-bias terms sit mostly in the landmark subspace. Layer 3 looks mixed: H0, H1, and H5 read as copy-like, while H2 and H3 are weak enough to call broad or unclear.

I'm working out whether L0H1 or L0H5 is the offset-1 head, since L0H5 already looks like the previous-token head that induction relies on. To test whether L0H1 is an offset-2 head, I'd compute a bilinear score between the L0H5 and L0H1 value-output paths on the L2 and L3 heads and check whether it comes out positive.

I'm checking whether L0H1 is an offset-2 head by testing the query's previous-token circuit against the key's prev-prev token, and I'll also look at L0H2 for the same pattern.

