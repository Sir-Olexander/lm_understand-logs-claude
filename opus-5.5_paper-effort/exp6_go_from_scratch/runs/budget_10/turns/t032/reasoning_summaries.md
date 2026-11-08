I'm checking the bias values for the output layer: the interior nodes come out to -1.6, which matches the offset from the negative weights once I account for the degree and τ. The weight for the edge from unit 488 to the PH output works out to -0.1 per PH unit.

Before launching the games, I want to check the rest of the logic, though I can't fully verify the empty-board logits by hand. The J offset is a worst-case figure, and that's acceptable. I'll run a games pass with 96 strong games and 32 weak ones, skipping random since the network already beats it every time.

