I'm looking at the top two singular components of the QK matrix in L1, where the keys seem to track the averaged-position code f(j) and the queries partly carry that same position signal. So the attention pattern is mostly set by the key's position code, weighted by the query, and I'm checking how the rank-2 score form q_i^T A f(j) fits that.

I'm working out what the query's raw position or MLP0 outputs might contribute, and how the query picks a direction in the f-plane so that keys whose averaged-position code aligns with it get the attention. I'm also considering whether, for 2D, f(j) traces a curve the query could point at relative to its own position.

Running averages make f(j) vary fast at small j and slowly at large j, so the keys can't give sharp local attention; the L1 heads more likely attend broadly, to early positions or to recent ones. I'm weighing that against the shuffled case, where what the model needs is a bag over the current line, so the query would have to pick out positions after the last newline.

I'm checking whether L0 heads that attend to newlines could give each query a positional code for its last newline, with L1 keys matching against it. H3's output is mostly positional, and H2's leans on token identity.

Next I'm checking the query side of those top L1 components, testing whether the queries pull from the output of H3 (its top few dimensions), the H2 and H0 outputs, and the raw position and token signals.

