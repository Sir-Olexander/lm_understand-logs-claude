I'm looking at how the Atari predictions break down by density. True ataris cluster at low density (0-3), while false ones show up more at density 3-7. At the higher density bands the false positives outnumber the true ones, but at density 3 and below the true ones dominate. So a density-based discount would cut false positives, though it would also suppress some real ataris.

I'm weighing whether to double-check the false counts, since for capture decisions at p the relevant figure is per (p, q) pair rather than per stone q. I'm also considering a different route: a coarser, leaky signal of nearby same-color stones with liberties, used only to discount capture bonuses rather than cancel them.

I'm weighing a density-weighted adjustment on the capture bonus: discounting opponent groups by their same-color neighborhood density, with the coefficient tuned so the existing units still carry the signal. The complication is that the same term would also dampen own-stone bonuses, and mixing opponent and own densities in one linear unit is messy, though it might still work.

I'm working out how to gate the capture-bonus unit so that healthy stones with no flags don't pick up a spurious positive value from the density offset. A plain linear relu with a constant shift would boost low-density stones, which is the wrong direction, so I need some way to keep the unit silent when no flags are set.

I'm weighing whether a cheaper way to extend reach by one or two hops exists, starting from the h1 and h2 stages of the current pipeline.

I'm looking at how a radius-2 seed rule might work, where a seed at i checks for an own stone two steps away connected through a middle stone. The problem is that each direction needs its own AND of the middle stone and the far stone, so it would take several units rather than one.

I'm weighing whether to drop the H layer and seed h1 with only the exact "own and e≥2" condition, but that would lose the pair seeds that matter in tight groups. Keeping H in h1 alongside the e≥2 seeds would let h2 build the pair seeds from H while still getting the single-liberty case right.

I'm checking whether the reach could grow through the h6 layer, by having the capture unit also subtract a propagated count from neighboring stones. The catch is that h5 would need to carry that propagated signal for opponent stones, and with only 26 spare units there isn't room for it.

I'm considering dropping T_own from h5 to make room for a propagated count G_opp, and then having the capture unit subtract that count so the A_opp part only fires when no nearby group is healthy. The trade-off is that the own-side escape signal and the SA/extension features would lose some information.

I'm trying to see whether the h6 budget can work if the capture unit reads a propagated count, but every split I try overflows the 567 units against the available space. Merging the POS and NEG units or folding J in might free enough room, though I'm not sure yet what that does to SA.

I'm weighing a density-based discount on the capture bonus, applied only to dense groups, so the adjustment lands in U rather than in the shared unit. The catch is that subtracting density inside U would also trim the atari and escape bonuses that should stay intact.

I'm considering splitting the capture unit into separate opponent and own-side versions, so the density discount only touches the opponent part, but that uses up the remaining budget in the output layer. To make room I'm thinking about merging the POS and NEG units into one.

I'm weighing whether a single relu unit per point can encode a one-sided territory signal, like penalizing own-territory moves while ignoring opponent territory, which would free up a unit. The open question is whether an asymmetric term like own influence minus a scaled opponent influence can carry enough information without a separate NEG unit.

I'm weighing whether to drop the NEG penalty to free up 81 units for splitting the capture unit, but that would let invasions go unpenalized. Dropping POS instead would let own-territory filling go unchecked, which wastes moves. Maybe I can compute POS and NEG over a shared representation to save space.

I'm considering merging the EYE and SA units into a single unit, since both draw on the same neighbor counts of own and opponent stones. The idea is that the eye term and the self-atari term could share one relu, though the scaling for the eye case might push the SA penalty in the wrong direction.

I'm trying to fold the eye and self-atari terms into one unit. Adding a scaled eye term to the self-atari pre-activation helps for true eyes, but two non-own neighbors would push the sum down by the full scale and wipe out the self-atari penalty, so I need a way to clamp the eye part at zero.

I'm circling back to the eye case: an eye point has no opponent neighbors and all its neighbors are own stones, and I'm checking whether the SA count can exploit that. The next line is cut off, so I'll leave off where that count was going.

I'm considering whether the eye case can be separated from a real connection by checking whether every neighbor is one of my own stones, since an eye has no opponent neighbors at all.

I'm trying to fold an eye check into the self-atari unit by using the count of opponent neighbors, since a true eye has none. The sum has to stay non-negative for real eyes, and adding a corrective term based on own-stone neighbors looks like it could work.

I'm working out whether a single relu unit can cover both the eye case and the self-atari case. The idea is to add a term that is positive only when the point has no empty and no opponent neighbors, which would mark a true eye.

I'm testing a constant of 10 for the eye term, which gives the right values for true eyes in both corner and edge positions. But when a lone point has one empty liberty and two opponent neighbors, the same term subtracts 10 and wipes out the self-atari signal that should still fire.

I'm ranking the per-point features by how much they matter: the tactical unit and self-atari are essential, and the eye term is essential for endgame. The influence terms POS and NEG are strategically important but have magnitude problems, and the linear J terms carry the open-area value and shape. I'm asking whether POS could absorb J's linear part through a relu with a large offset.

I'm asking whether the linear terms in J can be written as a sum over a fixed board kernel, which would let the same per-point units carry them. The catch is that the existing convolution only sees the difference between own and opponent stones, so the two colours can't be weighted separately.

I'm weighing whether POS and NEG can be replaced by a per-point influence term, so the territory signal lives entirely in the move being evaluated rather than in a summed board feature.

I'm wondering whether POS and NEG can just be computed per point from the move's own influence value, with no aggregation over neighbors. That keeps the penalty bounded, and then I could fold those self-features into J.

I'm checking whether dropping the NEG penalty and leaning on J's linear opponent weights could discourage deep invasions, but a linear term can't separate contact from deep intrusion. I'm also reconsidering the density discount for own groups, since an escape bonus on our own liberty would mostly waste a move rather than cause real harm, and the self-atari check already guards the worst case.

I'm concerned that the false captures are the real risk here: since the SA check is currently disabled by any opponent neighbor in atari, a move that captures can slip past it and get our stone taken. I'm considering letting SA count the liberties gained from captures instead, so the penalty would drop by one per adjacent atari opponent stone.

I'm considering a density discount on the capture bonus instead, since a true capture with no empty neighbours leaves our stone with a single liberty, and a larger captured group would give more room. The penalty from the self-atari check has to be weighed against that, so the discount should target captures specifically rather than the whole unit.

I'm weighing whether a combined linear term over the density of both colours can work for the capture bonus, but for an opponent stone the own-stone density around it is usually high, so mixing the two would wrongly shrink the capture bonus. That suggests I need to split the unit into separate opponent and own components.

I'm considering whether the open-area term in J could be dropped and its role taken over by the static prior and the POS/NEG self-penalties. The linear contact and shape terms might move into the U units instead, though the opponent and own density discounts would need to stay separate.

I'm looking at whether J's open-area base could go, since the static prior and phase units might already cover center versus edge preference in the opening. The contact bonus could move into the opponent-side units as a constant, but then the density discount has to stay separate there too.

I'm weighing whether J's terms (the own and opponent orthogonal weights, the diagonal and second-ring values, and the base) can go, since the U units could take over the contact bonus as a constant on the opponent side and the adjacency term on the own side.

I'm weighing whether J's contact and shape terms can be dropped in favor of the unit-based contact bonus, since losing the diagonal and jump terms could leave own stones loosely connected. Dropping NEG instead would keep J intact but leave invasions of the opponent's area unpenalized, and I'm not sure how often that happens.

I'm considering merging the eye check into the POS unit, adding a term that rises when most neighbours are already own stones, so a true eye would get a large penalty while ordinary own-area points would only get a mild one.

I'm reconsidering the POS unit with the eye term folded in, but a bonus tied to neighbouring own stones would wipe out the signal for most ordinary territory points, so that doesn't work. I'm now looking at whether the h5 features could feed U directly, though the A values there are only a ReLU of S minus the flags, which doesn't obviously give the same result.

I'm considering whether the density discount on the opponent-side terms could live inside the h5 layer itself, so A_opp would already be reduced by local density rather than needing a separate unit. The open question is whether a single relu over the stone signal minus the kill flags and a density penalty can still behave like the current A_opp, or whether the combined form breaks the two-liberty case.

I'm checking whether a density discount on the opponent-side term can be folded into the h5 layer, so A_opp is reduced by local opponent density. The problem is that for cells that aren't opponent stones, the expression could still go positive, so I'd need a gate that zeroes it out for them.

I'm working out whether the density discount can go directly into the opponent-atari unit at h5, so A_opp is reduced by nearby opponent density. A gate term would zero it out for cells that aren't opponent stones, so the expression can't go positive for them.

I'm working through whether a density discount can be built straight into the atari unit: taking the gain as 1 minus a small multiple of the local density, so isolated stones keep close to full weight. The worry is that some values would still come out positive for cells that aren't opponent stones, so the expression needs a gate to zero them out.

I'm considering a sharper threshold for the atari bonus, with a steep slope so it drops to zero above some density, but that gives isolated stones more than full weight. Taking a capture bonus scaled by that value would give small captures outsized credit. I'm weighing whether to cap it or combine it with the two-liberty check instead.

I'm now testing whether a group's far-reaching stones could be used as a seed to show two or more liberties, since two liberty-holding stones at distance three or more in the same group would rule out atari for the stone in question.

I'm checking whether the (r+2c) mod 5 colouring separates points within distance 2, since offsets like (2,0) and (0,1) seem to collide on the same colour and that would break the test.

I'm checking whether the mod-5 colouring helps here: points of the same colour are always at least three apart, so if two liberty-holding stones in a group share a colour, they'd guarantee two liberties. The catch is that the flags are only per colour class, so I can't tell whether I'm looking at one stone or two in the same class.

I'm checking whether a graded density discount can sit inside the opponent-atari unit itself, with a gate so non-opponent cells stay at zero. The next step is to see if reducing the SA kill by that graded value keeps the false-capture bypass in check.

I'm trying a graded version of A_opp: a density-scaled value that is 1 or more for low-density opponent stones, falls to 0.67 at density 4, 0.33 at density 5, and 0 at density 6. That would weaken the bonus on the 986 true ataris at density 4-5 and cut most of the 1,479 false ones at that range.

I'm sizing the capture bonus as a1 times the graded atari value plus a2 times the two-liberty term, with a1 set to 6 so isolated single-stone captures don't get outsized credit. That gives roughly 15 at zero density, 9 at density 3, 5 at density 5, and a floor of 3 beyond that. The falsely flagged far stones still keep the flat T bonus, which works like an atari-move bonus.

I'm now sizing the scale fix: a milder slope on the graded atari value, so the SA and EYE kills still trigger for isolated stones while dense false ataris get only a partial kill. The same K = 3 multiplier keeps the flagged far-stone terms dominant, since F is always an integer of at least 1.

I'm now setting the positional term: a self-only kernel keeps the penalty bounded, with the overall weight at τ = 2. For an empty point I'm building the influence map from the kernel weights, and for deep own territory that gives an influence of about 6, which should push the logit down to roughly -6.

I'm settling the base term for own territory: the empty-point count still gives a high base there, so the net comes out around -2, which isn't strongly negative compared with passing at -4. That means I'd keep filling my own territory, which is mostly harmless given the EYE and SA protection but still a weakness I want to note.

I'm settling on a penalty that sums the positional influence over a small neighborhood of the point, weighting the point itself fully and its neighbors at half, so deep own territory gets a strong negative value. That keeps the total bounded, which should be comparable to the SA and EYE kills, though I'm still concerned about the risk of overshooting.

I'm setting the pass penalty so it sits a little higher late in the game, around -3 at 60 moves. Using a slope of 0.15 gives about -6 at 40 moves, but at 70 moves it drops to -1.5, which might let pass beat a useful low-value boundary move. So I'm tightening the slope to 0.13, which leaves roughly -4.2 at 60, -2.9 at 70, and -2.25 at 75.

I'm settling the escape term for own stones: with the discounted value, an isolated stone in atari gets about 10 at zero density, falling to 6 at density 5 and 2 at density 10. Combined with the self-atari check, that still leaves the net negative whenever the escape doesn't actually work.

I'm working out how to implement the change with a scaled penalty, so that a deep-territory position lands near the pass value instead of overshooting, and I'm making sure the self-atari escape bonus keeps the net negative when the escape fails.

I'm settling the tuning values: graded atari bonus with the density term, capture weight 6 with a bonus of 3 for the two-liberty check, self-atari weight 15, and eye weight 30. For the positional term I'm summing the influence over the point and its neighbors with weights 1 and 0.5, using 0.25 for own territory and 0.2 for opponent territory. The pass penalty becomes -12 plus 0.13 per move.

I'm setting the atari-unit weights for own stones: the base weight becomes 1 plus a density term, the kill penalty is 3, and each nearby own stone at density reduces the weight by the same term. The opponent-side version follows the same pattern, using the opponent stone indices.

