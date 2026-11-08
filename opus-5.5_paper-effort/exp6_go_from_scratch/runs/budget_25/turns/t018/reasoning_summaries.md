Radius is what matters most here: at R=4 there are 831 false atari judgments, nearly as many as the 1045 true ones.

The false atari problem for big groups is the main concern, so I'm weighing whether group size can tell real ataris from false ones. A stone can't see past radius R, so it can't directly check whether its group continues beyond that. Counting reachable stones within R might flag big groups, but truly big groups in atari are also valuable, so that signal cuts both ways.

I'm sketching a way to get more reach per layer: a two-step jump that checks whether the next stone along a line is also in the same colour. But each such check costs a unit per directed edge, so it runs into a unit budget problem.

I'm now looking at a way to get past the depth-6 limit: give the opponent side radius 5 and read the atari signal linearly. For three colours, though, the own-side readout becomes nonlinear, so I'm checking whether summing the two non-matching colour counts can work instead.

I'm weighing whether a linear readout could work for the own side at R=5: summing the two non-matching colour counts gives a clean atari signal, but the sign flips for groups with all three colours present, which penalises extending strong groups. The false-positive count at R=5 for par2 is 588, against 476 for p3, so that trade-off is still open.

I'm working through the self-atari cases now. For the group that has no empty neighbours but whose neighbour has two liberties, a par2 atari readout at R=5 won't separate two liberties from three, so that case needs a K=3 check. The other case, one empty neighbour with the neighbour group already in atari, is covered by the par2 signal.

I'm weighing whether a cheaper signal could separate two liberties from three for the own side, since the par2 readout alone doesn't distinguish them. Filling a point with no empty neighbours is usually only worth it for connections or captures, so a moderate penalty for those moves might cover most of the cases.

I'm checking a linear version of the self-atari test at R=5: a penalty when a move has no empty neighbours, offset by a term that is positive for three-liberty groups. Summing the counts for each neighbouring group should give a rough estimate of how many liberties it has.

I'm checking the linear version of the self-atari penalty: it works for a single neighbouring group with three liberties, but a connection between two adjacent stones of the same strong group would get a bonus instead, which would wrongly reward filling own liberties.

I'm now trying to make the unit readout at L5 work for the own side: the NOT units would need to report absence cleanly, but gating them against garbage from non-own neighbours is still unresolved.

I'm checking whether a gated presence signal can be built linearly: the own-side indicator minus a relu of that indicator minus the summed unit values. For own neighbours this recovers the presence value, and for non-own neighbours it drops to zero, so they don't leak in.

I'm trying to build a linear version of the own-side gate: a unit that is relu of the own indicator minus the summed unit values, so that non-own neighbours drop to zero. The catch is that the own indicator has to be linear at that layer, and the three-colour ternary value isn't, so I'm checking whether shifting by b minus one can stand in for it.

I'm checking whether a gated presence unit can be built from the ternary colour value by shifting it, so that only own-coloured neighbours survive and empty or opponent points drop to zero. The shift needs a small margin so that empty points stay near zero, and with that margin the unit reads as absent-and-own exactly.

I'm checking whether the own-side presence can be made exact at layer five: a gated unit per colour that is one when the neighbour is own-coloured and absent, and zero for empty or opponent points. The presence value is then the own indicator minus that gated unit, which should leave the non-own neighbours at zero. That adds an own indicator layer with 81 units, on top of the 81 board units and the three gated units per point.

I'm checking whether a linear count of present neighbours can flag two or more own liberties, which works for the own side, but a separate test for three present doesn't come out non-negative. The simple difference formula lands at -1 for a single present neighbour, so I'll need another route for the three-liberty check.

I'm checking whether a clear separation between strong and weak neighbouring groups can be made, since merging a weak group into a strong one would be a good move that a linear penalty might wrongly flag as unsafe whenever the two counts happen to cancel out.

I'm trying to build a per-colour class unit that is one when all neighbouring groups are present, and the second class check is still being set up.

I'm trying to get a linear version of the "two or more present" check to work, and the cleanest route I see is a unit that is 2O_q minus the summed presence values, which is non-negative and only reaches zero for groups with at least two present neighbours. The catch is that this has to be computed one layer later, so the per-point unsafe check would need another layer to use it.

I'm trying to get a linear version of the "at least two own liberties" test to work at layer six, so I'm looking at a per-point unit that combines the own-side presence values. The aim is a term that drops to zero when a neighbouring group has enough presence, but that combination is still not settled.

I'm checking whether multiple neighbouring own groups break the exact test, since a sum over them can reach the threshold even when one group has three liberties. That case looks rare among true self-ataris, though it may matter more for safe moves that connect several groups.

I'm checking whether summing per-group signals works for the connection case: a weak group merging into a strong one could cancel out a strong group's surplus liberties, so the sum would wrongly mark a good move as unsafe.

I'm checking a proxy for the three-liberty test: summing each neighbouring own group's signal gives a rough unsafe flag, though it can cancel when a strong group and an atari group sit side by side.

I'm checking the one-neighbour-group case next, where a move is unsafe unless some neighbouring group has at least two liberties.

I'm working out the one-neighbour-group case: the unit for a move with a single empty neighbour should be one exactly when every neighbouring own group has just one liberty, and the sum of per-group presence signals gives that as a non-negative term. I'm now turning to the zero-empty-neighbour unit, which combines the same per-group signals with a large weight.

I'm checking the unsafe test for a move with no empty neighbours: the signal is the sum of each own group's presence difference, which can go negative for atari groups. That sum is only a proxy, so it can misjudge mixed cases where a strong group and an atari group sit side by side. I'm also making sure the presence values are reliable, since each adjacent own group should always have at least one present liberty.

Checking the escape units next: for each neighbouring own group, an escape signal should fire only when all of its colours are absent except the one the empty point would supply, which is a pairwise relu over the two absence units.

Now I'm working out the escape-related units at layer six, where a group counts as in atari when exactly one of its liberties is present. I'm weighing whether a per-colour split is needed, or whether a single relu over the presence sum for each neighbouring group is enough.

Now I'm checking the opponent side, where the radius limit forces a choice. Keeping the opponent units at radius five would take too many units at layer five, so I'm considering radius four or three, carrying each group's atari status forward as a single summed unit per neighbouring group.

I'm checking whether the own-side units can use the board channel at layer five instead of a separate own-indicator layer, since the difference between board and presence values goes wrong for opponent neighbours. The opponent contribution would then need a correction term, and I'm still working out how to supply it.

I'm weighing whether the empty-neighbour count can be read directly from the board and occupancy values at layer six, since the neighbour terms I was carrying through the earlier layers are running out of room.

I'm now checking whether the opponent side can fit within the layer budget at radius four, since the NOT units there would push layer five past its limit. I'm considering whether a single per-neighbour unit at layer five could carry the same information more cheaply.

I'm checking whether the opponent side can be carried in one unit per neighbouring group, since two separate zero-tests can't be merged into a single relu. The useful observation is that a group's colour always includes the parity of the empty point, so only the other colour needs to be checked for presence.

I'm noting that the opponent side only needs one unit per point, for the colour that is the parity of the neighbouring stones, since the other colour is always present next to an empty point. I'm now checking whether the propagation for the other colour can be dropped as well.

I'm noticing that own groups can skip the third colour in the atari count: for an odd stone, whose neighbours are all one colour, only the other two colours need checking. The same parity trick should work on the opponent side, so I'm trimming the readout to one unit per point and bringing the layer-five total down to about 486 units.

Now I'm checking whether the own-side and opponent-side units fit in layer five: with the opponent side at radius four, the total comes to about 486 units, and the escape and liberty counts go into layer six.

I'm checking whether dropping the board units at layer five would free enough room to push the opponent side out to radius five, since keeping both opponent channels there overflows the budget by about 15 units.

I'm now checking whether the opponent gating can come from the accumulator itself, since without b5 nothing else carries the empty-point signal at layer six.

I'm checking whether the opponent gating can work without the board values: the empty-point term could be carried by a relu over the opponent units, but the per-group gate still needs the empty-point term from the board values, so I'm weighing whether the non-opponent case can be left ungated.

I'm weighing whether the opponent gating can come from the opponent channels themselves: an opponent stone next to an empty point always shows the colour opposite its own parity, so that signal could stand in for an explicit gate. The obstacle is that a min over two signals isn't a single relu, so this route stalls.

I'm checking whether a normalised version of the opponent unit would work: a unit that is 1 when the opposite colour is absent next to an empty point, which would avoid the unbounded counts that an unnormalised version would produce.

Now I'm budgeting the first layer: the board values take 81 units, the own-group seeds take 243 at three per point, and the opponent seeds take 162 at two per point, leaving about 26 units spare.

Now I'm checking the layer budgets: the first four layers each come to about 486 units, leaving roughly 26 spare, and layer five lands near 446 with about 66 to spare, so the layer-six readout still has room for the local pattern units.

I'm now checking how the U0 proxy handles captures: the per-point penalty comes out of the empty-neighbour count and the signed neighbour terms, so only points with no empty neighbours get penalised. The capture bonus still needs to outweigh those penalties, and I'm also wondering whether a group with at least two liberties after merging counts as a real escape.

I'm weighing how the local pattern features at layer six should encode eyes and influence, with the positional part of the evaluation still to be worked out.

I'm sketching a cheap positional feature for each point: influence scores are linear, but turning them into nonlinear features costs units, and only about 66 units are left at layer five and roughly 107 at layer six, so I'm looking at what a contested-ness term like relu(1 - |I(p)|) could afford.

I'm considering adding the influence term directly to the output logit, which would be linear and need no extra units. The catch is that a stone's own gating would be disturbed unless the gate can tolerate that small nonzero term.

I'm weighing whether the output layer can read a linear influence term directly from the layer-six units, since that would avoid spending any new units, though stone points would need care so their gating isn't disturbed.

I'm checking whether the influence term can go straight into the output as a linear sum over the layer-six units: the capture bonus perturbs stone points only by a few thousandths, and empty points only by about 0.04, so the gating stays safe.

I'm weighing whether a linear influence term helps, since it would be cheap but may just push toward crowded or contested areas. Before picking a positional heuristic, I want to see whether the losses in the first set of games came from tactics or from territory, so I'll build the tactical part first and check what budget is left.

I'm now adding a term to the self-atari penalties in U1 and U0 that scales with the number of adjacent own stones, so that larger merged groups are penalised more heavily.

I'm checking the self-atari term in U1: adding a penalty that grows with the number of adjacent own stones, but then the two-liberty case still comes out positive, so the coefficient on the liberty count has to be larger.

I'm working out how to scale the self-atari penalty with the number of own neighbours, so that a larger merged group gets penalised more, while keeping the two-liberty case at zero.

Now I'm working out the opponent "not" unit at layer five, which depends on the neighbour terms across the group and its neighbourhood.

I'm checking the opponent "not" unit for the atari judgment: a group is flagged when its colour is absent from the local points, so a group next to an empty point counts as in atari. The own-stone term follows the same pattern, but with several colours it has to be taken as a sum of indicators.

Now I'm working out the own-stone unit counts: for odd points only the two colours other than the neighbours' fixed colour matter, so two units each, while even points need three.

I'm sizing the layer-six unit that counts own neighbours, and I'm checking how the nonlinearity lines up for the own-stone group.

Now I'm checking the zero-liberty case for U0: with no empty neighbours, a neighbouring atari stone makes the penalty term large and negative, so the self-atari penalty only fires when the move really is suicide-like.

I'm settling on a moderate coefficient for the self-atari term: with BIG at 10, a net negative count from atari neighbours pushes the penalty up to about 14 plus the own-neighbour count, while a balanced mix leaves it at about 4 plus that count.

Now I'm checking the self-atari penalty when a stone has no empty neighbours: with a neighbour that has two or more liberties, the BIG term has to be large enough to cancel U1, so I'm raising its coefficient to 20. If every neighbour is in atari, U1 stays at 12 plus the own-neighbour count, while U0 picks up the large atari term.

I'm weighing the self-atari penalty against the escape bonus for a single extension of an atari group. With the current coefficients, two adjacent stones from the same group would give a bonus that outweighs the penalty, so the balance needs adjusting.

I'm trying to make the self-atari penalty grow with group size without breaking the two-liberty case, which means the coefficient on the neighbour-stone count has to stay small relative to the steeper penalty on the liberty count. The escape bonus is the issue: a single extension could pick up more bonus from adjacent atari stones than the penalty costs, so I'm considering making the liberty term much steeper so the penalty doesn't scale with neighbour count while the escape bonus does.

I'm trying to build a per-point escape signal at layer six, one that fires only when an adjacent own stone is in atari and the merged group would be safe, instead of scaling a penalty with neighbour count.

I'm settling the escape bonus and the futile-extension penalty together: the bonus scales with adjacent atari stones, so the penalty has to outweigh it for up to three neighbours, which means the n_e term needs a steep coefficient and no scaling with neighbour count.

I'm settling the penalty weights for unsafe extensions: the futile-extension term has to beat the escape bonus even when a couple of adjacent stones are in atari, and a two-liberty neighbour group still counts as safe.

Now I'm checking the eye-fill case: an empty point whose neighbours are all own stones gets flagged, though that flag may not hold if the point is also an atari escape.

I'm settling the eye penalty against the escape bonus: if a neighbouring group is in atari and filling the point would connect it to a healthy group, that's a good move, so the escape bonus should beat the eye penalty when the merged group is safe. The unsafe cases are already handled by the other penalties, so I'm setting the eye penalty to 30 and treating genuine eyes of healthy groups as the only case where it fully applies.

I'm settling the pass threshold: the eye penalty of -40 and the self-atari penalties of about -26 both sit below a pass at -12, so those moves are only chosen when nothing better exists. The open question is whether the policy should pass when only moves that fill our own territory remain.

I'm checking whether a 2-point eye counts as eye space: playing the first point is safe while the group keeps the other point and outside liberties, but once those outside liberties are gone the group would be left with a single eye and die.

I'm checking the 2-point eye case again: filling one point of a straight three-point eye leaves two single-point eyes, so the group survives, while filling an end point leaves a two-point eye that doesn't. Since the bot will fill its own interior points once the board settles, that's harmless under area scoring.

I'm wondering whether repetition could become a problem if our invasions keep getting captured, though the board usually changes enough that it shouldn't loop. I'll come back to the pass-1 repetition losses later and check the endgame, where the opponent passes and our invading stones keep going.

Now I'm laying out the positional side: a simple line-based value plus a mild influence term, with an eye-pattern feature on top of the tactical ones.

Now I'm working out the first-layer seeds for own stones: each stone gets a gated term that only fires when it is occupied and its neighbours contain enough empty points, so an empty point can't trigger it by accident.

Still working out the two-liberty and three-liberty conditions for the even-parity stones, with the weights on same-colour neighbours.

Now I'm checking the neighbour-count thresholds for the even-parity stones: two-liberty and three-liberty gates, with the weights on the same-colour neighbours.

Still working through the first-layer seeds, now for the opponent's parity-2 stones: the gated terms for the libs of each parity, with the same two-liberty and three-liberty thresholds.

Writing the propagation layers now: each one passes activations to a stone and its neighbours, with a gate that suppresses them unless the right conditions hold, and I'm setting the gate bounds per layer.

I'm checking the big-constant gates for the propagation layers: M of 1000 is comfortably above the largest pre-gate sums at every depth, so the gating holds, and the NOT-style units stay non-positive whenever their gate is active.

Now I'm setting up the layer-five features: the own-stone indicator is a simple threshold on the layer-four activation, and the opponent indicator stays zero.

I'm checking the L5 open-liberty and opponent indicators: each one comes out as an exact 0 or 1 because the gates force everything else to zero. That gives a count of 446 units at this layer. Now I'm starting L6, where the board-point activation picks up small linear contributions from the L5 units.

Now I'm defining the L6 board unit so that its activation picks up a small offset plus a capture-related term from the neighbouring L5 units, which keeps empty points at exactly 1 plus a tiny perturbation.

Still writing the eye and influence units...

Still writing the logit formula...

Now I'm weighing whether the output should use b6 for the capture signal or pass the n_q values through L6 directly.

I'm settling on the output layer: the capture signal can ride through L6 by folding it into the board unit with a tiny weight, so I don't need to spend extra units passing the neighbour counts through. Then I'm adding a small positional term that scales with the L6 perturbations near stones, so moves close to existing stones get a mild preference.

Now I'm weighing a positional term: every stone's occupancy is linear in its own and its neighbours' values, so the output can read the board directly, which means passing O6 through as another set of 81 units.

I'm sketching a version 2 of the network and want to check it against positions where I can compare the actual activations with what the circuit should produce. For the positional side, I'm leaning toward the line values from version 1, plus a linear proximity term that rewards contact with stones on the board.

I'm settling on a positional term: the line value plus weights for own and opponent stones at distance one and two, leaning toward contact moves and near-centre play in the opening, and I still need to pick the exact weights.

I'm settling on the positional term now: rather than spending the 26 spare units, I'll keep the line values and add weights for stones at distance one and two, then test the version 2 network against positions with known activations.

