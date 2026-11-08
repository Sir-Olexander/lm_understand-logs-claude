I've decided token recovery won't work because the forms aren't clean enough, so I'm dropping it. Before settling, I want to check whether the Python-versus-English difference could be explained by newline-relative position. If a head attends strongly to a newline-like token from a constant query, its key-bias vector should be sharply peaked, but the L1 heads' key-bias vectors vary and I can't yet judge how peaked they are.

I'm weighing whether the sparsity of MLP0 neurons reading the token signal helps, since the space token shows up far more often in Python than newlines do. I'm also going back over the induction-head argument for the Python model. The L1 heads show only weak positive induction traces, so the evidence for stronger induction capacity in Python is modest, which nudges things toward English.

I'm weighing the probabilities for English, Python, and romaji, landing around 0.55, 0.25, and 0.20, and I'm keeping the ordering as it stands at about 0.92. Then I'm labeling the L0 heads: H0 looks local, with a positional QK pattern and token-copying values; H1 and H2 look broad, with rank-1 positional attention that writes to a positional summary; H3 is the previous-token head.

Now I'm going through the rest of the heads: L0H3 is the previous-token head, L0H4 looks broad like H1 and H2 but weaker, and L0H5 has nearly zero attention scores and writes a strong position signal into coordinate 21 that the L1 and L2 heads read as a key bias.

Now I'm going through the remaining heads in layers 2 and 3: the L2 heads attend locally but more broadly, with L2H3 the weakest and L2H4 showing only a faint induction-like pattern. The L3 heads H0, H1, H2, and H5 look broad, writing into shared non-token output directions, while L3H3 stands out as the copy head, matching the query token against the previous-token signal from L0H3.

Let me double-check the key-bias figures for layers 2 and 3 against the list, and then settle how the final answer should be organized.

