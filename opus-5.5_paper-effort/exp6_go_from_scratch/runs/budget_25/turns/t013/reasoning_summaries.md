I'm weighing what the strong bot does differently: it plays contact moves that build connected shapes, while ours land scattered on the third and fourth lines. So I need a better sense of position, covering territory, connectedness, and local responses. I'm also thinking about which policy features could produce sensible territorial play, and I'm looking at influence functions from classical Go programs, where each stone spreads influence and territory is wherever that influence clearly favors one side.

I'm now weighing a simple local heuristic for the weak-versus-weak case: favor moves next to existing stones that extend our own liberties and cut into the opponent's, while also preferring empty points where the surrounding area is contested. I'm also considering how to judge the final result, which comes down to whose walls enclose more territory, and I'm leaning toward estimating ownership for each empty point from the nearest stone color.

I'm settling on a Voronoi-style frontier heuristic: for each empty point, compare its distance to our nearest stone against the opponent's, so contact moves at the boundary, where both distances are small, look most valuable. Moves that push into the opponent's side while staying close to our own stones are the ones worth preferring. I'm also wondering whether a network can approximate these distance features with a few dilation-style layers, though the liberty propagation may compete with them.

I'm now treating influence as linear in the inputs: each point's influence is a distance-weighted sum of stone signals, so a couple of nonlinear units could turn it into a balance score. I'm also considering keeping own and opponent influence separate, so the value of a move depends on how much it shifts ownership of the nearby area.

I'm working out a two-layer scheme: the first layer computes a peaked "balance" value for each neighboring point, using a few ReLU units to form a triangle around zero influence difference, and the second layer sums those values weighted by their distance to the candidate move.

I'm checking whether the influence signal can be computed directly from the board passthrough, since each point's influence is linear in the stone values. The worry is that adding the accumulator's output into empty points would add noise to that signal, so I'd need to keep that contribution small.

I'm settling on an influence-based way to score a move, combining tactics, territory gain from a triangle-shaped function around the candidate point, and a shape term. On an empty board the influence is zero everywhere, so territory gain just favors points with more neighbors, which pushes the choice toward the center. Once stones are on the board, the neighborhoods should start to differentiate the candidates.

Now I'm checking whether the influence-based gain approximates the Voronoi frontier well enough, since placing a stone at p raises our influence on nearby points and the value depends on how steeply ownership changes there. The triangle-shaped weighting should reward contested areas, but I'm wary that it could push the bot toward loose moves that a stronger opponent cuts through, so tactical checks like atari and capture still need to carry weight.

Next I'm looking at the game records to see why we lose so many stones: we average 23 lost against 5 captured. My working guess is that our larger groups are getting caught in atari and not saved, so I want to count, move by move, whether one of our groups was already in atari when we played.

I'm going to tally the game records next: for each move, whether one of our groups of two or more stones was already in atari, whether we could have saved it by extending, whether an opponent group could have been captured, and whether we played a self-atari, using my Go engine on each board.

