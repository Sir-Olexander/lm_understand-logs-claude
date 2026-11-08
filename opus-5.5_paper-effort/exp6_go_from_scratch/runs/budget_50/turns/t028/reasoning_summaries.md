I'm noticing the opponent makes contact moves next to our stones in most of their turns, while our own moves are mostly isolated or pure extensions. That suggests the strong bot fights at contact, pressing our liberties. So I should answer contact in kind, with hane, extensions, or blocks. Next I'll measure how often the opponent's moves touch ours.

I'm weighing a "fighting" heuristic that values moves next to opponent stones, giving extra weight when those stones are short of liberties. I'm also considering a liberty-race view: reward moves that add liberties to our weak strings and cut the opponent's, measured on the local stone level.

I'm weighing whether string-level liberty counts beyond two are reachable, since the current propagation only certifies "two or more." A local stone-level proxy might approximate the 3 and 4 classes, but I'd need to free up budget, and the Voronoi term is the likeliest place to get it.

I'm weighing what really decides games at this level: keeping groups alive, capturing the opponent's weak stones, and managing liberties in fights. I'm also checking whether the losses come from territory or from walls that split our stones apart. A "connectedness" term that rewards linking our own stones, plus a cutting term against the opponent, might address that, along with a contact-response rule.

I'm weighing a way to score each candidate move by the liberties it adds to our stones and removes from the opponent's, using per-stone counts within a small radius instead of full string counts.

I'm thinking about getting more accurate string-level liberty counts through a better propagation scheme, which would mean dropping the Voronoi term to free up capacity. Pseudo-liberty sums are a common trick in fast Go engines, but summing over a connected string doesn't propagate exactly through a ReLU network, so I'm considering max or OR propagations instead, which can certify "3 or more liberties" locally.

I'm working out how to fit a "≥3 liberties" class for both colors into the propagation budget. Dropping the Voronoi term frees some capacity, but a full class for both colors won't fit in the middle layers, so I'm considering a ≥3 class for our strings only, or a shorter propagation radius.

Next I'm checking whether the losses come from big groups dying, comparing captured stones in lost games against won ones.

I'm counting how many of our stones and the opponent's were captured at once in each game, to see whether we're losing whole groups.

