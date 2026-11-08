I'm tracing the top-left black group in game 34, which gets captured by white's move at (1,2) after black connects at (0,3) around ply 58, and I'm checking the board rows at that point to see which stones belong to the group.

I'm checking the liberties of the black group around (1,2) after ply 58, and seeing that filling (0,3) leaves the group with only one liberty, so white's capture follows. I'm also weighing whether (0,3) looks like an eye, since its neighbours are all black, but the atari on (0,4) would undo that eye status.

I'm working out the score for the black group's remaining liberties and getting about -7 after the eye and escape adjustments. That doesn't match the policy's 0.9993 for (0,3), so I suspect every other move is scoring even worse, maybe through the territory penalty or the pass value.

I'm checking the board around columns 6-8 and the bottom to see if the empty space there explains why (0,3) scores so far above the other candidate moves. I'm weighing the atari on (0,4) and the liberties of the neighbouring black stones to work out where the large bonus for (0,3) comes from.

I'm working through the score for (0,3): an eye-check against the atari on (0,4), which gives roughly -7 before the positional terms. Since the policy's 0.9993 is far above that, I'm wondering whether every other candidate scores below -15.

I'm thinking that the other candidate moves were all scoring badly because they sit in territory that one side already controls, so the penalties drag them down toward -10 or -15. That leaves the tactical move at (0,3), around -7, as the least negative option, and the real problem is that it's a self-atari move.

I'm concluding that -7 was the best of a bad set, since every other move scored even lower, and that the pass logit around -7.6 sat just under it. That points to the positional penalties being too large next to the self-atari penalty. I'm considering bounding the positional values, say to [-4, +6], and making the SA penalty much steeper so tactical blunders always fall well below neutral moves.

I'm noting that the positional term grows without bound, since it sums over many empty points inside solid territory and can reach around -40. I want to cap it, either by saturating each point's contribution or by redefining the unit so its maximum stays near -6.

I'm trying to pin down a bounded version of the positional term, since summing over neighbouring points makes it blow up, and I'm considering limiting the penalty to the point itself and its immediate neighbours.

I'm weighing a saturating form for the positional term, since bounding each point's contribution would need two units per point, which is more than the budget allows. So I'm looking for a single bounded function of the influence value that still keeps deep-territory moves near zero rather than far negative.

I'm looking at a way to bound the contribution of each point with a trapezoid-shaped penalty, which would need four units, and then I'm considering whether two units per point using a clamped difference could approximate it.

I'm now looking at whether the positional terms can be bounded at all, since the trapezoid shape would need four units per point and I only have two. I'm considering a single relu on each side of the influence value, with the linear empty-point term carrying the rest, but the sum still goes unbounded as the influence grows, so I'd need to cap the influence itself.

I'm considering making the self-atari penalty much steeper, say around -30 per unit, so that a blunder like (0,3) falls well below any merely useless move, and I'm weighing whether that alone is enough to make the pass look better than a tactical mistake.

I'm checking whether the huge self-atari penalty could misfire on positions where a group still has a shared liberty or a neighbour with three or more liberties, since that would wrongly block good moves.

I'm settling on a scoring weight of 15 for the self-atari penalty, so the escape bonus of 9 leaves a move like (0,3) at about -6. That's not much lower than the pass, and if the escape still leaves the group in atari the move is useless. I'm also turning to the positional penalties, aiming for deep own territory and deep opponent territory both near -6, boundary points near +2, and open area near +4, by reducing τ·γ.

I'm going with a smaller kernel for the penalties, only the point itself and its neighbours, so the positional sum stays bounded. That should put deep-territory moves near -6 and keep self-atari below them.

I'm weighing a two-layer approach to saturate the influence, but that would need more units than I have, so I'm considering a count-based version instead. I'm now trying to rework the rest of this with a much smaller kernel, so the penalty stays bounded.

I'm weighing whether to drop the negative-side penalty altogether, since invasions into opponent territory would then score as open area, but that might only cost tempo, which matters less in the endgame. I'm also considering a symmetric saturating unit that treats large influence in either direction the same way.

I'm trying to see whether the positional term can be folded into the existing units, so the linear board terms ride along in the stone-level units instead of needing their own unit for each point.

I'm considering whether a unit per stone can carry the linear board terms: each own or opponent stone's unit would feed its neighbours, but status effects could leak to points two steps away, which might actually be useful for atari situations.

I'm weighing whether to keep the linear board terms in the per-stone units, since they might leak status to points two steps away, which could be useful in atari situations.

I'm checking whether the positional value can be floored with a single relu over a linear pre-activation, but the influence penalties depend on nonlinear POS and NEG terms, so the sum would need another layer to bound it. I'm now considering computing POS and NEG at the h5 level instead.

I'm considering whether the positional penalty can be floored by making it a linear board proxy instead of the nonlinear influence product, so a single relu in the output unit bounds it from below.

I'm testing whether a single relu over a linear board proxy could floor the positional term, but the dominance-versus-contest distinction needs the nonlinear POS and NEG units, so that route doesn't work directly.

I'm narrowing the positional penalty to just the point itself and its immediate neighbours, weighted by distance, so the total stays bounded. That should keep the influence-based penalty within a range the budget can handle.

I'm working through a realistic estimate of the positional penalty for territory points, using a 3x3 own-territory example where the surrounding wall stones contribute to the influence sum. That puts the influence around 6.6 at the center, so territory points land near 6 in penalty, and I'm about to total that across the five-point kernel.

I'm working out the penalty scale: with a positional weight of about 0.2 and a bound near -8 for deep territory, the boundary points come out only around -1.4. The open-area base term runs from roughly 6 at the center down to about 4 at the edge, and the self-atari penalties sit at -15 for one liberty and -30 for two.

I'm checking the numbers for the escape move, which comes to about -6 once the self-atari penalty is applied. Deep territory lands near -4, roughly level with passing, so it doesn't matter much there. What matters more is that self-atari and eye-filling stay well below passing, while captures still outweigh the positional range.

I'm weighing whether a single liberty-weighted contact bonus could capture the value of attacking an opponent group with few liberties, but the budget only covers up to two liberties, so reaching three would need another feature. I'm also considering a modest boost to the extension bonus for own stones with two or fewer liberties, and I'm wondering what to do about ladders, where the net keeps running from atari.

I'm moving on to the question of the false-atari and repetition problems, but first want to fix the positional scaling, since that's what led to the self-atari choices in game 34. To check how often that happened, I'm planning to count the plies where a move of ours was captured on the opponent's very next move across the pass-4 games.

I'm working out how to measure the self-atari blunders, counting moves where our stone got captured on the opponent's very next turn across the pass-4 games.

