I'm looking at which words the relative output change measure flags: proper nouns, color words, names like Isabel and Logan, plus 'Asia', 'six' and 'cinq'. Response words are where I'm headed next.

I'm reading through which answer types show up: colors, places, and person names like Charles Dickens, so the dataset may be general trivia with short entity answers. Since the output vocabulary is so varied, the copying has to work for whatever the user said. The identity test showed no difference, so the sycophancy probably comes from attention shifts in the query-key projections of a few layers rather than from the output side.

I'm now planning to decompose the change in the attention score for each head into a residual-space matrix and pull out its top singular directions, mapping those back to tokens to see which query and key features shifted. The sign matters here: a positive singular value means attention rises when the query and key projections point the same way along that pair of directions. Since RoPE rotates dimension pairs by position, I'm going to split the analysis into low-frequency and high-frequency parts, treating the low-frequency part as content matching that holds across distances.

I'm going to split the query-key change into low-frequency and high-frequency rotary dimensions and look at the low-frequency part, which should carry content matching that holds across distance. Then I'll take the singular decomposition of that difference for the relevant layers and heads and map the top directions back to tokens.

I'm checking how the rotary frequency pairs contribute to the query and key changes, computing the squared norm of the changes per pair across heads and comparing it against the baseline query and key norms in each layer.

