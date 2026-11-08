I'm sorting through the corner position, where the black stones have a capture-and-recapture cycle going. Each side keeps taking stones back and forth, and the sequence ends up back where it started, so I'm checking whether it repeats and whether our moves gain from the capture bonus.

I'm now thinking through the bottom-right corner, which seems to follow the same capture-and-recapture pattern. These cycles involve only one or two stones in dead areas, and the position can repeat because the engine can't tell what happened earlier. To reduce this, I'm considering penalizing captures that leave our capturing stone with just one liberty, since that makes it easy for the opponent to recapture.

I'm weighing a change to the capture scoring: counting each captured stone once instead of doubling adjacent stones, so a single-stone ko capture gets penalized. The catch is that a large capture through one adjacent stone would look as bad as a ko, and I can't measure the size of the captured string directly.

Checking the top-left corner again, I see the self-atari: after the opponent's (1,0) move, playing (0,0) leaves our two stones with only one liberty at (0,2), and I still can't see why the 0.997 score favored it, unless the white group's atari flag is pushing the capture bonus too high.

I'm checking why the (0,0) move scored so high: the white stone at (1,0) has one liberty, but its group sits far from any liberties, so the atari flag may be firing falsely and boosting the capture bonus.

I'm weighing whether to deepen the propagation, or to let the certificates reach farther through a wider neighborhood sum, though that might count stones from separate strings as connected. A false "safe" reading would hide real ataris, and I'm trying to judge which of the two mistakes costs more.

I'm considering a different way to fix the false atari flags on large groups: instead of relying on the flag, I could propagate certificates across two-step jumps along straight lines, though that adds per-direction units and gets expensive.

I'm wondering whether the big-group atari problem could be handled by checking how far a string extends, rather than relying on the flag alone. A large string with liberties far away would be recognized as safe, while a short string gets the full atari penalty.

I'm thinking about speeding up propagation by letting each layer reach farther, perhaps by precomputing which same-colored neighbors are directly connected. Pairing each point with its right and down neighbors would give a fixed set of connectivity units, which might let certificates spread two steps at once.

I'm weighing a fix for the big-group false atari problem by propagating certificates through a wider neighborhood, but that risks leaking across unconnected strings. I'm also considering pair-connectivity units to spread certificates further per layer, though that needs another nonlinear step to combine them, and I'm not yet sure which mitigation is worth the cost.

I'm weighing whether the spare channel budget could be used to extend the propagation, but carrying values forward costs the same per point as propagating them, so that doesn't save anything. I'm also thinking about making the first layer's certificate stronger so that large groups get certified across more positions.

I'm considering two-step jumps per layer to reach distant liberties faster, since the group's nearest liberties sit about seven steps away and no local certificate can cover that gap.

I'm trying to get certificates to reach distant liberties by doubling the propagation step, so each layer covers two steps at once. Taking the matrix-power view, combining a two-step path needs an AND of two binary values for each pair of points, which means far too many units.

I'm considering a cheaper way to tell whether a string is fully explored within radius 3, by tracking whether any stone of the same string sits exactly at that distance, so large groups can be flagged without another propagation pass.

I'm weighing whether to accept the false atari flags as a cost, since shifting certificates would mean more propagation units. I'm also considering whether big-group false flags could be dampened by checking the liberty frontier near the candidate point, but that doesn't yet tell me which outcome matters most.

I'm considering a simpler fix for the repetition: lowering the per-neighbor capture weight from 2 to 1. That would push false captures without real support below the penalty threshold, while healthy captures with several supporting stones would still score well, though a true ko capture would also get avoided.

I'm going to lower the repetition-prone capture weight and make the U2 penalty relative. Then I'll look at whether black's weaker results (31% versus 69% as white) come down to komi 7.5, and check how close the losses are to the strong baseline and what kinds of losses they are.

