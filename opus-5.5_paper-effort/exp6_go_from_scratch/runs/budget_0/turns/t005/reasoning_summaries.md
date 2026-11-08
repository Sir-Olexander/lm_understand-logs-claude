I'm finding the interval bounds too loose, which is why the S minimum comes out at -1790. I'll switch to tighter semantic ranges, treating the binary features as {0,1} and SOFT as [0,2], and then look at the negative weights in S.

Checking the bounds on the SOFT unit: the subtracted terms are all non-negative, and the nested relu keeps NV2 at or below own, so SOFT tops out at 2. The rest of the S-side positivity argument holds under the semantic ranges too.

Now I'm checking the unit definitions against the semantic ranges, starting with the L1 units. The Eown unit works out to E_q less 5 plus 5 times own_q, and the Sown unit holds up as already verified.

I'm checking the remaining unit definitions: the VS gating bound comes out negative when A1 is 0, so that case is safe, and the L3 through L5 units (U1, NV1, U2, U3own, U3opp, NV3opp) all hold as expected.

I'm going through the S-side weights for the next group of units: the SOFT term carries a -50 weight, and I'm checking the L6 contributions and the diagonal and jump terms, plus the EYE, IN, and ALLOPP checks, and they all look right.

Now I'm weighing whether the pass move is safe: if every legal move is worse, the pass logit at -20 beats the in-territory moves at -130, so passing dominates. I'm also considering the case where the opponent keeps playing inside my territory and I keep passing.

I'm now looking at a possible pass-related edge case: the opponent may fill points just outside my group's liberties, which could leave it unsafe, so I'm considering how the ESC and related units handle that. I'm also checking shared-liberty seki, where filling a point both groups share would count as self-atari and get penalized if there's at most one empty neighbor. Superko doesn't need attention here. Finally I'm weighing that in crowded endgames, opponent groups' liberties next to my walls are dame, and filling one is worth a point in area scoring.

I'm weighing the ladder case: a stone in atari that keeps extending into two liberties would trigger the escape term each time, so the logic would run the chain toward the edge. I'm accepting that as a known weakness, since the opponent may not chase the ladder well. I'm also starting to think about the case where the opponent's group next to mine is unsafe.

I'm thinking about whether contact moves are overvalued: lots of my groups end up with only two or three liberties, so the opponent can keep ataring them. Maybe I should weight diagonal and jump connections higher so my stones form solid walls, and lean on the ATK term at +12 to keep the pressure up.

I'm considering lowering the contact bonus to +3, so attaching to a strong opponent group is roughly neutral, while contact with a weak group gets the large attack bonus. I'm also weighing a small plus for adjacency to my own stones, offset by the lost liberty, since fights tend to hinge on those blocking moves.

I'm thinking about whether to swap the NV3opp feature for something mixed, though I'm leaning toward keeping it. Reducing the plane carrying in L5 doesn't look useful either, since radius-3 U propagation costs the same as carrying U2.

Now I'm weighing the weights once more, especially the prior and relative scales, along with how to handle eyes and two-point eyes. Points in my territory next to my stones count as IN and I leave them alone, while the quiet points away from my stones are worth playing. Once only empties touching my stones remain, the leftover regions stay as eye space, and if the opponent plays inside one of them, I'll see that play as a threat to my group.

I'm checking whether passing is ever a problem: the pass logit is far above the in-territory moves, and if the opponent keeps playing into my space I'll respond to any stone that lands next to my territory. I'm also thinking that a stone the opponent plays inside my 2-point region is in self-atari, so capturing it is fine, though it's worth noting that leaving it alone would hand them those points.

I'm weighing the HARD penalty on opponent territory throw-ins, where a lone stone with one empty neighbor should lose to passing, and that looks right. Next I'm checking the end-game case where the opponent's group has a single eye, so an all-opponent-neighbor eye point is a legal capture worth the big bonus. I'm also looking at a group with one eye plus one outside liberty, and wondering whether the outside liberty counts as safe enough to flag.

I'm checking whether the group with one eye and one outside liberty counts as safe, and whether the stones around that eye form any seed through their connections. The eye-wall stones don't link up directly, so the eye alone gives no seed, and I'm weighing whether the outside liberty can provide one instead.

I'm checking whether a stone next to the eye forms a seed with a neighbor, which would make the group safe, and if not, weighing the attack bonus on p. If the wall beside p isn't very safe, the SOFT penalty would stop me from playing there.

I'm settling the contact bonus at +4, which leaves a net +1 over an empty neighbor, and keeping ATK at +12 for attacking the opponent's stones. I'm also giving unsafe opponent stones a +80 capture bonus.

Next I'm checking the "IN" exclusion, making sure points adjacent to my own stones at distance 3 aren't wrongly marked as in-territory. I'm also considering whether to exclude points next to weak own groups, though that seems unnecessary since low liberties usually come with nearby opponent stones.

I'm going back over the scoring weights, checking the prior and relative scales, and deciding how the eye and two-point-eye cases should be handled. Once that's settled, I want to spot-check some of the saved weight matrices by mapping the nonzero entries back to their named features.

Checking the named weights for a few units: the 'Sown' feature in L2 at (0,0), the 'S' feature in L6 at (4,4), and 'IN' at (0,0).

