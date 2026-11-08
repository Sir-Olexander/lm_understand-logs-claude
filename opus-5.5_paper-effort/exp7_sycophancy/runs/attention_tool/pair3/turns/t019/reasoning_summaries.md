I'm seeing modest selectivity overall, with most composition ratios landing around 1 to 1.7. The clearer signal is that the query side of several layer 10 and 11 heads reads outputs from layer 5 heads at roughly 1.4 to 1.75 times, while L10h2 leans more on layer 1 heads and a4.3.

I have two runs left, and I'm weighing whether to spend one on a final consolidated ranking of heads. I think I already have enough data for that. The MLP side is less settled: I'm wondering whether the late layers (L15 through L17) matter most for digit-aligned changes, or whether mid-layer MLPs carry more of the arithmetic. My plan is one last quick run that compares absolute digit-logit changes per MLP layer and per attention head, which should settle the ranking question.

