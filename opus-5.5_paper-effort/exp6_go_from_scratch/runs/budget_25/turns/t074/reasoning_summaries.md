I'm looking at the self-ataris in both games: each one plays beside a large opposing group of 18 or 36 stones whose liberties are being weighed, and the move seems driven by a capture-judgment bonus on those big groups. In game 9, for example, the stone at (2,0) sits next to an 18-stone group that still has 4 liberties.

I'm checking whether the group's liberties at (0,8), (2,8) and (3,8) matter for the capture judgment, but those are eight columns away and fall outside the visible radius, so only (2,0) registers. That makes big groups look like false positives, and I'm considering how to reduce that without a rule that just ignores group size.

I'm working out whether a walk-count channel over the opponent's group could act as a size signal, since a single stone would stay at 1 while a large group would grow exponentially with its degree. The catch is that an extra opponent channel through all five layers would overflow the budget, so I'm looking at reusing the existing one instead.

I'm working through whether the big-group false positives could be reduced using the existing opponent channel, since its value at each stone reflects liberties of the same or opposite parity. The check is whether, within the radius, no stone of the relevant parity has liberties, which would tell me the group's far extent is simply invisible.

I'm checking whether the capture bonus could be reduced when a stone has two opposing neighbours, since that is where the false positives seem to come from. Looking at (3,0) in game 9, it touches one opposing stone at (4,0) and another at (3,1), so the sizes of the surrounding groups would be the deciding factor.

I'm checking whether the 1-liberty opponent stones next to (2,0) could be told apart from a true small group by how many same-colour neighbours they have, but that looks weak. I'm also wondering whether the n4 suppression of the capture bonus should be dropped in favour of some other signal, since after a true capture the move would have liberties of its own.

I'm weighing the cost of the Uc penalty on true captures: with no own neighbours and no 2-liberty support, the penalty outweighs the capture bonus, so I'm considering a partial suppression of the n4 term instead of removing it outright.

I'm weighing whether a larger threshold on the capture penalty could offset the bonus, but that would also block genuine captures. I'm running the numbers on a partial suppression of the n4 term, where a true capture's penalty stays below the bonus while the false positives still get suppressed. Checking the L5 channel budget is the constraint that keeps pulling me back.

I'm working out the suppression by computing it inside the Uc unit with v5 linearly, since n5 would need L6. The added term would un-suppress the penalty whenever the par(q) colour shows up at radius 5, which should matter most where opposing stones have no same-parity presence within R4.

I'm thinking about whether the false-capture case could be handled by how far the opposing group's liberties extend: if I play beside a large group that only has two liberties, playing one of them puts the group in atari, which is a reasonable move anyway.

I'm weighing whether suppressing the capture bonus could depend on whether the capture would really free up liberties for my own group, since that would be true for real captures but not for false positives on big groups. Small opposing groups seem checkable within radius 5, but large ones don't, so I'm wondering whether group size itself could serve as that signal.

I'm testing whether the opponent channel at (3,0) in game 9 can serve as a size proxy, and I'm starting to tally its value there: parity-1 stones with a liberty seed channel 0, and the stones at (3,0) and (2,1) are the only ones contributing.

I'm now considering whether the suppression could instead be limited to cases where the move itself would gain liberties, since a true capture frees the adjacent points while a false positive on a large group does not.

I'm considering making the suppression depend on how many captured stones actually border the move, since a real capture always frees at least one adjacent point, while a false positive on a large group does not.

I'm checking whether the L6 budget can absorb the eye unit's contents, since P1 is only a linear carrier and freeing its 81 units would let n4 also reach L6.

I'm looking at whether the P1 unit could be folded into P2 by rewriting the positional value as a single relu, but the two-relu forms don't fit either, and folding it into Uc or the n5 units doesn't work since the output only reads n5 at neighbouring stones.

I'm checking whether the per-stone counts for the opponent groups could be reduced: the own-side quantity that matters is just the sum of those counts per stone, so the question is whether it can be computed with fewer separate relu units.

I'm now checking whether the per-stone counts of the opponent groups can be merged, since the only own-side quantity that matters is the sum of those counts for each stone. The budget is already full, so I'm weighing whether the positional term could be estimated from something cheaper than the current opponent-stone units.

I'm now considering whether the escape bonus could be folded into the existing Uc penalty with a sign flip, since at_q is only relu(S_q - 1) per own stone and the per-point sum does not fit cleanly.

I'm questioning whether the false-capture problem justifies restructuring the network, given the current 77% strength. Before deciding, I want to see how the losses differ from the wins, especially in terms of tactical errors like false-positive captures.

