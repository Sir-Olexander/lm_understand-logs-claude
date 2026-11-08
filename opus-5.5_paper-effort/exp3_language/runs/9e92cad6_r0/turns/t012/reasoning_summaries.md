The first approach is turning out to be muddy, since the positional and token directions only partly separate. I'm going to look for a cleaner way to split them, starting from how the input is built as the sum of token and positional embeddings.

I'm going to test whether later heads read the current token through the L0H3 positional mover. I'll compute the T-restricted matrix for each head and compare its trace against a random baseline.

