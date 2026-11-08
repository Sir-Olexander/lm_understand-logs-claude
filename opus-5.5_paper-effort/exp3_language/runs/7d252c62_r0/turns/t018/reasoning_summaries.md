The top direction in the Wfc input has little projection onto T (0.21), so MLP0's main direction must be something else, perhaps position, the attention-output residual, or the mean direction. That didn't settle it, so I need a better test for English versus Python. Python indentation looks like a good angle: the leading spaces are kept, and lines start at 0, 4, 8, 12, or 16 spaces, so there are long runs of consecutive spaces. In the shuffled condition those spaces would be scattered, which should show up as a difference.

I'm looking for indentation as a way to separate Python from English. Counting the runs of spaces could come from recent-window attention in the early layers, or from a head that attends a fixed four positions back. Some later heads also seem to write a dominant direction, so I'll check those too.

I'm weighing whether the newline token could be the cleaner signal for telling Python from English, since Python lines end and start with indentation while English paragraphs start with capitals. Without identifying that token directly, though, I'd need to find it some other way, perhaps through a line-boundary direction in an early head.

I'm considering the space token, since it's the most frequent one in both English and Python. The top singular direction of the previous-token OV might capture the main variance, maybe the space-versus-letter contrast. I'm also thinking about the query-independent key bias in L1, which looks very large, and I want to check whether L0H0 and L0H4 output directions drive it.

Projecting L1's key preference onto the L0H4 and L0H0 output directions should show which one drives it: the position average or the token-specific signal. I'm also thinking the first position may act as an attention sink, since there's no BOS token, and L0H4's uniform averaging would make position 0 stand out. That seems like a general pattern rather than something specific to English or Python.

I'm thinking the copy-heavy heads in layers 2 and 3 might be a clue, since Python is far more repetitive than English. The space token is likely very common in Python too, with a lot of leading-space runs, so I'll check how the space value vector shapes what comes after a space.

I'm wondering whether the doubled-letter patterns could separate English from Python, since both have plenty of them, though Romaji adds its own doubled consonants. The previous-token heads in layers 0 are purely positional, so they shouldn't carry any token-specific signal there.

I'm reading the L0H3 positional eigenvalues as negative, which would push it away from nearby keys toward distant ones, and its token-specific block suggests it might pick out particular earlier tokens, possibly the earliest newline. For L0H2 the positive positional eigenvalues point to a local, recency-favoring head, and I'm checking how its key-token preference combines with that.

I'm checking whether indentation runs or the newline token could separate Python from English, but I'm still settling on the induction-style copying that would track indentation across lines.

I'm working out how L0H0 could track the count of spaces since the last newline. If it attends to every newline evenly, the result is an average position rather than the most recent one. The positional eigenvalues in its query-key block might add recency, which would sharpen attention onto the latest newline.

I'm weighing whether the uniform average in L0H4 adds anything beyond the positional embedding already in the residual stream, since position 0 would stand out and larger positions would converge toward the mean of all positional features. I'm also trying to work out what L0H0's query-key structure does in positional terms, though that's hard to pin down without the positional embeddings themselves.

I'm thinking about whether a 2D-uniform positional head like L0H4, with its large output norm, would matter for a corpus where position within the line counts. Python lines are short, so knowing how far back the last newline was could help. Maybe the line-start keywords in Python versus capitalized sentence starts in English would be a cleaner way to tell them apart, though I'd still want to identify tokens directly through the induction head's output.

I'm trying to pin down which token each copy head reads, since the output directions of several copy heads should all promote the same token. A neuron's response on a token would then be a linear read of that token's direction, so I'd need to check whether the neuron weights line up with the token directions at all.

I'm still weighing whether space-counting heads in layer 1 could track indentation depth in Python, with the nearest non-space token acting as a proxy for distance back to the last newline.

I'm now asking whether L1's heads that attend to the most recent non-space token could track indentation run length in Python, using position-carrying values. I'll check how much L1's value reads come from the positional signal versus the L0 head outputs.

Next I'm weighing the likely design of this puzzle: six conditions across three corpora and two orders, so the solver is expected to read the answer off the weights. For the English-versus-Python split, indentation and newline frequency stand out as candidate cues, with positional heads that track line starts. I'm also considering whether the repeated-space bigram could help. Then I want to check the L0H3 and L0H0 token-keyed heads, asking which token each one keys on and how often it appears.

I'm estimating the token frequency spectrum from the singular value decay, since the stacked singular values may not match how common each token actually is. The L0H0 key direction is strong on most dimensions, so the token it keys on probably isn't a rare one, unless it's a class of tokens.

I'm checking the letter frequencies in Python code against English, since 'e', 's', and 't' dominate both and the distributions may not differ much. The "self" and "s e l f" bigrams don't seem usable as a discriminator. I'm also looking at how the runs of spaces in the code might separate the two.

I'm considering whether a head attending to the most recent newline could be useful in Python, since line starts drive many predictions there, while English has few newlines and would gain less from it. I want to measure each L1–L3 head's preference for newline keys in the query-independent term.

I'm now checking which heads carry a strong, query-independent preference for particular key tokens, since a head with a rank-1 bias-driven key preference would behave like an "attend to token X" head. I'll measure that bias-driven key-token preference for every head and compare it against the token-dependent QK norms.

