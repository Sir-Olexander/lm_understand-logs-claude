The captures fell off sharply, from 41 per game down to 8, once the scap term went to zero. That means many real captures happen where a stone has zero or one empty neighbor, so the penalty was throwing them out. Win rate fell to about 50% too, which tells me captures are central to winning, since killing opponent stones is where the wins come from. I'm settling on captures as the key thing to get right.

I'm going back to the pass 7 setup with the scap penalty at 2, since that's the config that held a 50% win rate against the strong opponent and 94% against the weak one. Captures are clearly where most of the wins come from, with 41 captured per game against 18 lost in that run. Next I want to add capture-focused tweaks, like atari moves and chasing ladders, and handle the false big-group flag without bringing the penalty back down to zero.

I'm weighing whether the repetition losses, which show up in only a few percent of games, are worth fixing by making us less willing to play inside the opponent's territory late in the game. Since a false big-group flag mostly costs us a stone that was already in their area, I'm checking when the repetitions actually happen, around moves 80 to 120, and whether the opponent just recaptures each time.

I'm turning over a way to tell real atari from false big-group flags, testing whether the string's only liberty is the point in question. The idea is that a stone next to p with another empty neighbor would already be safe, so it shouldn't be counted against us. I'm also considering whether a wider look at the local shape could catch the false flags before they cause the self-atari blunders.

I'm considering dropping the ≥3 certificate class, which costs about 162 units per layer, and spending that width on a radius-2 certificate in the early layers instead.

I'm considering a radius-2 version of the certificate: using pairwise connectivity features from layer 1 so layer 2 can judge whether a string has two or more liberties within a wider neighborhood. The catch is that the evidence for neighboring stones isn't available to other layer-2 units, so I'd have to threshold the layer-1 liberty features directly instead.

I'm considering whether the propagation could reach further by letting each layer pass along two-hop information, since the current N steps only see about one hop of certificate evidence per layer.

I'm checking whether two-hop ANDs could grow the certificate radius faster, but the intermediate cell would already have been certified in the previous layer, so nothing new comes from it. The radius only grows by one per layer, so a two-hop term would need fresh certificates from two steps back.

I'm working through whether two-hop certificate terms could widen the radius faster than one step per layer, but the intermediate certificate would already exist from the previous layer, so the two-hop version only adds reach that the normal propagation already gives.

I'm noting that a precomputed pairwise feature could give the AND needed for two-hop terms, but any AND on fresh certificates still costs a layer, so propagation stays at one hop per layer. I'll try expressing the AND as a threshold over the linear sum instead.

I'm checking whether a gate on the intermediate stone could be folded into a single threshold, but the AND of two different stones doesn't reduce to one linear unit in general. So I'm accepting the propagation limit of one hop per layer and moving on to look for efficiency gains elsewhere, since the strong bot seems to lose many stones to our captures.

I'm moving to the pass 7 losses to look for repeated blunder types. My plan is to replay each loss, find the first big capture of our stones, and check whether we played into self-atari or left two-liberty groups undefended. Since 96 games only gives about ±5% noise, I'll want larger samples of 128 games against the strong bot before trusting any subtle change.

I'm planning to count our atari strings that could have been saved by extending or capturing the attacker, and checking whether we actually saved them. I also want to tally our own self-ataris.

