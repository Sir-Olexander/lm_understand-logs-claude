I'm noticing the key-token preferences of heads L0H0, L0H2, and L0H3 line up with u, the densest MLP0 token, so u seems to be a boundary marker that these positional heads attend to, probably a space or a newline.

I'm checking whether the "current u, prev u" neurons really carry a double-space or blank-line signal. Only 8 of the 33 have a strong previous-token alignment with u, and in English double spaces basically never occur, so that prev-token weight may be unconstrained by the data.

I'm weighing whether the b·u component of these neurons is really constrained by the data, since "space then space" almost never shows up in English. The cosine between the current and previous reads is fairly high, around 0.6 to 0.79, so the overlap might be a genuine signal rather than leftover weight. I'm also wondering whether u is simply the one-sided "is space" axis, given only one neuron points strongly the other way.

I'm testing whether u is a newline rather than a space: if so, L0H3 would be tracking the column position of the last line break, and the densest MLP0 group would be the case where the current token is a newline, which fits Python code.

I'm reconsidering the doubled-u cluster: if u were a space, the doubled neurons would be detecting space-space, which almost never occurs in English, so either u is something else (a newline, which would fit Python blank lines) or the doubled cluster is an artifact. The strong centroid match between the doubled cluster and the densest token makes the artifact explanation less likely, so I'm leaning toward the Python reading.

I'm going to check whether the doubling shows up across all six top peak directions or only for u. If every token has some doubled neurons, it's just a generic effect from common letter pairs like ll and ss; if only u stands out, that points to whitespace doubling.

