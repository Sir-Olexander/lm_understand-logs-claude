I'm checking the sign of P1 at empty and own-occupied positions: at empty squares it works out to η·n_e, which is fine, and at own stones it's large but masked out anyway. The remaining question is keeping P1 positive at opponent positions, which the relu takes care of.

I'm worried the huge logits at occupied points, around 4e4, could cause trouble in float32, but they should be fine as long as the masking sets them to -inf or subtracts a large constant. Capping them isn't necessary. Next I'm planning to run the games, with 96 for the strong setting and 32 for the weak one.

