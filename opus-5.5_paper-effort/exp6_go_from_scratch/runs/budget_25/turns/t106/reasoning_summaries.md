I'm checking the opening: the first move was not the center, and the first-move probabilities look fairly flat. From there, the stones gather in the middle while White surrounds them on the left and top, and our contested-ness heuristic seems to push the play into central contact fights.

I'm weighing ways to fix our positional play, mainly toward territory and away from center clustering. One option is shifting the line-value weights so lines 3 and 4 beat the center, and maybe trading some contact-fight preference for open space. Our fights do win games, though, and as black we likely need more territory, so I'd want to test that carefully. I could also use the color plane to run different positional settings for black and white.

I'm thinking about what actually limits us: the opponent plays positionally well when it doesn't blunder, so beating it reliably means better positional judgment, maybe through retuning the line values and contact-fight weights. Its tactical blunders are what produce most of our wins, though, so more aggressive play that creates atari chances could be a better lever than pure positional tuning, and I'm considering how to detect two-liberty groups that a move could reduce to one.

I'm working out how to detect a single stone that has exactly two liberties, so a move could reduce it to one and put it in atari. Locally that means counting its empty neighbors and checking for no same-color neighbors, which the board layers might let me express in one unit.

I'm working out how to tell when a single stone has exactly two liberties, since that's the case where a move could leave it in atari. It comes down to counting its empty neighbors against its same-color neighbors, which I'd need to express as a single unit.

I'm working out a unit that flags single stones in atari or with two liberties: if an opposing stone has no same-color neighbors and enough of its neighbors are occupied by my stones, a relu-style term can separate the two cases, giving a stronger signal for one liberty left than for two.

I'm weighing whether the budget allows a per-stone two-liberty detector across the whole board, and it doesn't: 81 units would leave only 22 spare, so that's out. I'm wondering if it's better to swap the positional nonlinearity for it instead. A cheaper option is black-specific positional tuning through the color plane, which would need only one passthrough unit, and I'm leaning toward testing that.

I'm working out how to make the line values differ by color: a white-to-move indicator unit can shift each point's output weight, so black and white can get different per-point preferences. The center and line weights could be split that way. Lambda and mu can't be split the same way, since they sit inside the nonlinear units.

I'm considering whether an opening-book bonus for tengen as black could work: a first-move-only term that fires on the empty board. The catch is that the layer it needs is already full, so I'd have to drop a few points from the n4c layer to make room for the extra unit.

I'm now looking at whether our passing logic is costing us points in close black losses, since dame might be left on the board when we pass. My plan is to check the final positions of those games to see how many neutral points remain and whether we passed first.

