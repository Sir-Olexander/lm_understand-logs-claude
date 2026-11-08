I've found that 525 neurons mostly read from the T subspace, with about 81% of their projected weight there. The clusters aren't clean per-token groups, so the neurons aren't simple single-token detectors. I'm going to try the dual-basis angle instead, looking at a neuron's preactivation for each token as a dot product with its embedding.

Now I'm weighing whether the dominant direction in T separates a single token from the rest or a whole group like vowels, and I'm looking at how the neurons line up along it.

I'm weighing whether the dominant T-direction tracks a single token against a larger group, but I can't tell from the neuron weights alone. Next I'm considering the third OV direction of head L0H5, which may read a bag of characters in the window, and whether it matters for detecting Python. Indentation runs after newlines are another angle, since they produce repeated space tokens in multiples of four.

Now I'm working out how the attention could match each indentation position to the previous line's column, using a query and key built from column offsets. That would need a quadratic score on the column code, and the diagonal weights on coordinates 21 and 99 might encode the column as a rotation, though I can't confirm that without more information.

Now I'm looking at whether the newline token carries more weight than the space token in the segment structure, starting with the big structural direction in W_proj.3.

I'm weighing whether dir 1 is a "space" logit and dir 22 a "newline" logit, since newline should be a strongly predicted token in Python and rare in English, which would make its direction weaker in English-heavy data. Dir 22 has moderate strength but only a few neurons, so that's a loose fit at best. I'll also check whether the induction copy map can label the letter-copying directions.

I'm weighing English as the most likely corpus, then Python, then romaji, and I'm reconsidering whether the Python-versus-romaji order holds. The single late induction head is a small point against Python, but the heavy word-segmentation machinery fits every candidate. I'm checking whether the L1 heads that avoid repeating the same token would work differently for romaji, where the previous token carries a lot of information.

Next I'm thinking about whether the early positional heads need many dimensions to encode column offsets up to 80, since English only needs short word-position ranges, and Python's indentation and line structure would demand more precise positional information.

I'm checking whether the recency pattern of the early heads L0H1, L0H2 and L0H4 would be stronger for newlines in Python than for spaces in English, since both would need to pick out the last occurrence in the window.

I'm trying to reconstruct the positional embeddings from L0H5's output, reading coordinate 21 as a position code, since a recency signal would need its value to shift steadily with token index. Then I'm going to test whether L0H3 acts as a one-step shift and treat the positional embeddings as a linear recurrence, with each position being a fixed rotation of the one before.

I'm wondering whether the corpus could be told apart by how often newlines appear in the context, since Python would show many and English almost none. I can't label the third OV direction of L0H5 that way without more evidence, though. I'm also thinking that romaji text is so regular that local n-gram heads and MLPs would handle most of the work, which would make induction matter less.

I'm leaning toward English at about 0.45, with Python at 0.3 and romaji at 0.25, since the strong L3 induction head is worth weighing, and I'm not yet sure whether the vocabulary breadth points more toward English.

Now I'm estimating how many robust dimensions each candidate corpus would need: romaji has roughly 24 tokens above the frequency threshold, English around 25 to 26, and Python around 28.

Comparing the expected counts of robust dimensions, romaji (24 to 25) falls short of the observed 26, so it looks less likely than English or Python. English and Python both fit the count, and the 26 lower bound doesn't separate them yet. The single late induction head is a weak point against Python, but it isn't decisive either.

I'm settling on English as the most likely corpus at about 0.5, with Python around 0.28 and romaji around 0.22. The ordering stays roughly as it is. Before I commit, I want to check whether a shuffled-lines hypothesis changes anything: if the model learned on text with characters permuted within each line, would the prev-token and induction heads still form in the same way?

Now I'm checking the L0 heads, starting with H0, which uses a positional query-key score while its value reads token identity, with a mix of real-positive and complex eigenvalues.

I'm leaning toward labeling the L0 heads H1, H2, and H4 as "broad" heads: they seem to attend to a particular token type anywhere in the context, though they may also weight recent positions. H3 looks like a previous-token head, and H0 is local-ish. I'm not fully sure about H1, so I'll keep "unclear" as a fallback.

Now I'm sorting the L1 and L2 heads: the L1 heads look local, since their large key bias on coordinate 21 favors recent positions. For L2 the recency bias is weaker, so I'm leaning local for most of them, with H3 as broad and H4 as a possible weak copy head.

I'm checking whether coordinate 21 actually tracks position, since the large key bias on it would only make sense if its value changes steadily along the sequence. If the coordinate peaks at the first position, these layer-one heads might be drawn to the start of the context whenever the content doesn't override that, so I'd need to know the sign before reading them as recency heads.

I'm checking whether the sign of coordinate 21 changes with position, since that decides whether the L1 heads are truly recency-driven or pulled toward the start of the context. I'd need to trace how the position signal from L0H5 relates to the rotation pattern I found in L0H3.

I'm working out whether the slowest direction in the position embeddings is monotonic in position, which would make the uniform L0H5 head's output track position. For classification I'm treating L1 heads as local and L2 as mostly local, with possibly a broad one, and at L3 I'm labeling H0, H1, H2 and H5 as broad, H3 as copy, and H4 as broad with weak QK evidence. I still want to look more closely at the L2 heads' recency bias.

I'm considering whether the space token could be identified structurally, by checking if its direction falls in the null space of the induction copy map, which would mark it as a formatting token. I'm also thinking about telling English from Python by the bigrams around the most frequent token, since Python lines often start with indentation after a newline.

Now I'm checking whether the newline-then-indentation pattern shows up as a direct mapping in the first MLP layer, which would hint at Python-like text. Without labeled data I can't confirm that yet, so I'll leave it open and look at how the induction head behaves on runs of spaces.

I'm checking whether the space token's direction in the copy map is near zero, which would show it rarely gets copied. I'm also planning to compare the L0 segment heads' key directions against FC0's top direction to see if they line up with a space detector.

I'm going to check whether the L0 segment heads' key directions line up with the top FC0 direction and the dominant neuron-cluster direction. Alignment would suggest the segment token is the main token feature, which fits English or Python better than romaji, while misalignment would point toward romaji or a newline segment.

