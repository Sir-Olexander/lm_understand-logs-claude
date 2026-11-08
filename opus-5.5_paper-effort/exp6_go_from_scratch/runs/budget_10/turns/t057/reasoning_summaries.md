I'm checking the J offset: it's the negated sum of the minimum of own and opponent linear terms, clipped at zero, plus one, and the J unit adds it back. I'm also checking the worst case for J pre, where each point counts as own, opponent, or empty, and then I'm moving on to the linear influence term for the neighborhood of p.

Now I'm checking the unit that fires when three of the four pattern conditions hold with the fourth point empty, which should give the intended bonus at the cut or connect point.

I'm weighing whether the +3 connect or cut bonus is worth it when the diagonal stones are already joined elsewhere, since then the move would be redundant. I also want to check the J weights, now that the linear influence term sits inside J, and the scale of the POS weight at τ = 2. Before running games, I'd like to inspect a few positions, especially cut/connect cases and endgame territory, to see how the logit landscape looks.

I'm planning a positions pass before any more games, since the new L6 weights could break a games run. I'll inspect around 40 positions, drawn from earlier games at various plies, including the cut and connect cases, plus some late-game positions and the empty board, to check the activations.

