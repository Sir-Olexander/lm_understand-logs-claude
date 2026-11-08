The data shows hundreds of cases where a threatened own group of two or more stones could have been saved and wasn't detected, plus many self-ataris of such groups, so group-level tactics clearly matter most. That means v2 needs group liberty counting, with the quality of the own side (avoiding self-ataris and finding saves) weighted heavily. I'm now starting to work out the design of v2 in detail, beginning with the propagation steps.

I'm weighing a channel budget for the propagation layers: the board representation plus separate channels for each side's colorings would overflow the budget, so I'm looking at whether to reduce one side's channel count or merge the accumulator into the board channel.

I'm checking whether the gating survives a small perturbation: if own stones store slightly less than 2, the subtraction eats only a sliver of each positive signal, so positives stay positive and zeros stay zero. The catch is that the next threshold assumes exact values of at least 1, so I'd need to see whether a signal dropping to about 0.9 still gets read correctly.

I'm weighing whether the budget forces a trade-off: using a shared channel or an uneven split between sides. Keeping the opponent side at two classes might save room, but it would leave me unable to spot two-liberty enemy groups to attack deliberately, even though it only costs some false positives on atari detection.

I'm working out how to fit the channels in budget: mod-3 coloring for both sides would overflow it, so I'm weighing how to split the channels. I'm also considering whether the board representation itself could absorb one of the channels, but that breaks the gating, and I'm stuck on how to encode neighbor counts without circularity.

I'm trying to make the propagation work without the neighbor count getting in the way: each own stone could carry a constant offset, so the sum over neighbors picks up how many own neighbors a point has. I'm checking whether that count can be folded into the value itself so the gating still works.

I'm checking whether the board channel can also carry the opponent's coloring, but gating on the board breaks if its values shift. Neighbor-count offsets don't help either, since each stone's base would depend on second-order neighbors. So the board channel has to stay separate in every layer from L1 through L5.

I'm checking whether skipping the gating in some layers would be safe. Points without a stone would pick up colors from every neighboring group, so a point between two own groups would leak one group's liberty colors into the other, which is the same kind of mixing I was trying to avoid.

I'm settling on the propagation depth and the channel split: with the board channel kept separate, I can fit either three classes for the own side and two for the opponent, or the reverse. Gating has to run in every layer, since skipping it would let one group's liberty colors leak into a neighboring group, so the total budget is roughly five color channels across both sides.

I'm weighing whether the opponent side can get by with two channels, since three for each side plus the board overflows the budget. Two channels would keep the atari detection cheap but blind me to deliberate attacks on two-liberty enemy groups. I'm also looking for a way to get multi-liberty certification out of two channels, because the layer widths don't fit otherwise.

I'm considering whether the opponent side could get by with two channels, since that would cost some atari false positives but leave room for three classes on my own side, where self-ataris matter a lot. I'm also wondering if the own-side self-atari check could avoid needing exact liberty counts for neighboring groups.

I'm checking whether a self-atari at p can be decided without exact liberty counts for each neighboring group. If a group touching p has any color other than p's own, it keeps a liberty besides p, so the only hard case is a group whose colors are all p's color.

I'm checking whether a self-atari at p can be decided from colors alone: if the adjacent groups and empty neighbors carry any color other than p's own, a second liberty exists, and only the case where every color matches p stays ambiguous.

I'm checking whether a self-atari at p can be decided from the colors around it alone: if the neighboring empty points and adjacent own groups show a color other than p's own, a second liberty exists, so I'd only need to track, for each of the two other colors, whether it appears among those neighbors.

I'm settling on whether a self-atari at an empty point can be read straight from the colors around it. The neighboring own groups and empty points only need to show which of the two other colors appear, so if only one shows up, the merged group has just one liberty, and that check becomes a per-point step rather than a full liberty count. The point's own color channel is useless there, since it would only mark the point itself.

I'm settling on the opponent's capture check: a group adjacent to p is in atari there only if its colors are all p's color, so that check stays per-group rather than merged. For my own side, the same merged-liberty test at an empty point p covers both avoiding self-ataris and judging whether an escape from atari actually gains liberties.

I'm checking the own-side merged-liberty test at an empty point: an escape counts only if the merged group keeps at least two liberties, which the colors can show. Parity coloring looks useless for that, since p's neighbors all share one parity, though other liberties of the adjacent group might still carry p's parity.

I'm weighing whether the merged-group safety check at an empty point can be decided from colors alone, and whether the capture check against an adjacent opponent group stays per-group. Both checks look like they need a test that compares how each option would actually behave on real positions from strong games, so I plan to measure decision accuracy for the capture case and the self-atari case directly rather than keep reasoning about it abstractly.

I'm laying out the layer plan for the merged-group check: it needs one extra step beyond the propagation, so the empty-point color presence falls out of step-3 neighbor information, effectively a radius-4 read.

I'm working out how to tell whether a point is safe for my own group: the neighboring empty points would need a count of which colors they carry, and I could seed that at the first layer and carry it through the propagation steps without gating at empty points.

I'm working out whether the own side can get the extra empty-point information it needs, since the own-side propagation is switched off at empty points and a separate unit would be required. I'm considering computing the neighbor color counts at the last layer with a NOT unit, but the empty-neighbor counts don't come out linear in the activations, so that route is in doubt.

I'm checking whether the own side can get the empty-neighbor color information it needs, since its propagation is gated off at empty points. One idea is to let own-channel units at empty points carry the adjacent groups' colors instead of zeroing them, though that garbage would then have to be kept from flowing back into stones.

I'm noticing that a stone's pre-activation can't tell whether a neighbor unit is empty, since the weights are fixed, so letting empty points carry garbage doesn't work. Instead I'm considering a sixth layer that reads the fifth-layer no-reach units at the own stones around an empty point, though non-own neighbors would add noise.

I'm checking whether the own-side merged-group test can be read from colors at an empty point. Non-own neighbors would add garbage values, so a plain sum over them doesn't work. I'm considering a positive-polarity unit with a clamp instead, but that is just an affine rewrite of the same NOT unit, so the garbage remains.

I'm working through how to check whether a stone group's merged liberties at an empty point can be read from colors alone. The neighbor-color counts need to pass through the empty-point gating, and the non-own neighbors add garbage that breaks a plain sum, so I'm weighing a positive-polarity unit that is zero at non-own stones.

I'm checking whether the empty-neighbor color info needed at p can be obtained at the later layer, since passing it through every layer would be too costly. Reading it off the propagated activations at the empty point doesn't work directly, because the needed term isn't monotone in those values. So I'm looking for a different way to fold the empty-neighbor information into the own-channel propagation.

I'm looking at whether the own-side merged check at an empty point can work at the final layer: a sum over the neighboring stones' own-channel units, plus a term for empty neighbors by color. That empty-neighbor term needs the empty-point status from earlier layers, which I don't have yet.

I'm checking whether the empty neighbors of p can contribute their color info through the own channels without leaking into the stone updates. Since the weights are static, a value seeded at an empty point would still mix into adjacent stones, so this route looks unworkable as stated.

I'm reconsidering the empty-point garbage idea: if empty points carry tiny values (around 1e-3) while real stone signals are at least 1, a threshold in the propagation could filter the garbage out. The catch is that real values lose the threshold amount each step, so the scaling would have to compensate.

I'm checking whether the own-side empty-point info can get through the propagation without mixing with the big garbage from neighboring stones, and it can't, since the tiny values sit in the same sum. So I'm looking at a different gating for empty points, but that's not resolved yet. Next I'll try reading the empty-neighbor information from the opponent channels, though those are zero at empty points.

I'm weighing whether to stash a small offset in the b unit at empty points, so that 1 + ε·z carries the neighbor-color info through the gating. The idea works for the own and opponent gates, since a tiny deviation at empty points doesn't break their strong negative push. The open problem is reading z back out cleanly, because at occupied points the same subtraction would give a huge spurious value.

I'm checking whether the empty-neighbor color counts can be packed into the small offset on the bias unit at empty points, so the readout only needs one scalar. The open question is how to encode two separate counts in that single value without a nonlinearity to decode them.

I'm working out the merged-safety test at an empty point p: counting the own-side presence from neighboring groups plus the empty neighbors of each liberty color. The idea is that p counts as safe when both colors are present, or when two empty neighbors share the same color.

I'm now setting up the L6 readout at empty points: a negated presence unit per liberty color, built from the L5 presence values at neighboring stones minus an empty-neighbor term that still needs its own encoding.

I'm still working out how to get the empty-neighbor color counts into the NOT unit at L5/L6, since a single encoded value in b mixes the two colors together. Summing the [b_e == 1] indicators over the neighbors of each color looks like the right term to add.

I'm pinning down what "present" means at an empty point p: a liberty color k counts as present if some neighboring empty point has color k, or if an own neighbor's group has a color-k liberty other than p.

I'm now asking whether an own stone's group has a liberty of its own color, and I'm treating the empty-neighbor test as the sticking point. Each neighbor either contributes an empty point of its color or an own group's presence, and I'll try building a single unit from the b-channel values for that.

I'm checking whether the empty-neighbor colors can be carried per empty point at all, since passing two separate counts through every layer would be too costly. Reading the opponent-side signals to cancel the opponent neighbors' contribution doesn't hold up, because those signals aren't reliably positive at every opponent stone.

I'm working out how to get a single number per empty point that can carry the neighbor color counts, since the merged-safety test needs them. I'm checking whether an unsafe-point indicator can be built as a NOT unit over the neighbor counts once that encoding is in place.

I'm trying to write the unsafe indicator as a disjunction of two parts, one per color, where each part needs the presence check for that color and a test that the other color has fewer than two empty neighbors. The sticking point is that the check for at least two empty neighbors of one color is itself not a simple linear threshold, so I can't express it directly with the units I have.

I need the two unit definitions to separate the two colors: one where the big weight sits on the other color's empty-neighbor count, and one with the roles swapped.

I'm checking whether one encoded number per empty point, z = e_+ + 3·e_-, can carry both neighbor-color counts. The trouble is the thresholds: the value 0 and 3 would need to pass one test while 1 and 2 must fail it, and a single relu can't separate a middle band like that.

I'm checking whether one combined value per empty point, z = 3·e_+ + e_-, can separate the two unsafe cases. The first unit works out: with no own presence in the + color, z ≤ 1 captures exactly the situations where the + side is unsafe, and a relu of (2 − z) minus a large weight on the + presence sum gives the right sign.

I'm checking whether a single encoded number per empty point can carry both neighbor-color counts, and it can't, since no one threshold separates the cases each unsafe check needs. So I'm considering whether the opponent-side channels, which sit at zero on empty points, could carry that extra information instead.

I'm weighing whether the opponent-side channels, which sit at zero on empty points, could carry the second neighbor-color count, since one encoded number per empty point can't serve both unsafe checks. The worry is that garbage written there at empty points would leak into opponent stones on the next layer. So I'm considering a separate route for the merged-safety test that avoids that contamination.

I'm considering whether the merged group's liberties can be checked by treating the empty point as a virtual own stone, but propagating from it would leak into neighboring units. I'm reconsidering the approach by seeding from each own stone with an empty neighbor, though that could be too many cases to handle. Instead I'm focusing on three decisions: whether a move captures an adjacent opponent group in atari, whether an own group in atari can escape, and whether a move would be a self-atari.

I'm working through when a move at an empty point gets a merged group safe: if it touches two or more empty neighbors, or an own neighbor group already has three or more liberties, or a single empty neighbor plus an own group with another liberty, or two own groups each with two liberties and different outside liberties. The first case could come straight from the local neighbor count already in the z-encoding.

I'm checking the cases where the merged group's safety is ambiguous, i.e. when the best neighbor group has only two liberties. One common case is an own group with two liberties that share an adjacent point with p, so playing p would leave it with one. Another is an own group in atari at p whose other neighbor has two liberties not adjacent to p, which I'm still weighing.

I'm estimating the merged group's liberties from the neighbor counts, using a lower bound that takes the larger of the empty-neighbor count and the best own group's liberties minus one. The upper side would add the empty neighbors to the sum of the own groups' liberties, which gives a rough approximation rather than an exact count.

I'm trying to see whether a merged group's safety can be read from the neighbor counts, and the open question is whether the accumulator can take the extra information if it is kept separate from the z-encoding.

I'm weighing whether the accumulator could carry the e_- count, but adding it there would leak into the output logit, and the output layer has no clean way to subtract it back out. So I'm considering changing how the e-values are used altogether.

I'm settling on a coloring where every neighbor of p shares one parity, so the empty neighbors of p all carry the same color and a single count can stand in for them. With that, the only color that needs the neighbor-count encoding is the one opposite p's parity, and I'm now checking whether own neighbor groups can cover the remaining case when the merged-group check runs.

I'm checking whether a coloring where all of p's neighbors share one color would let a single count stand in for the empty neighbors, since the color-c(p) presence check fails because p itself can be a liberty of that color. Using mod-3 coloring, the neighbors of p have colors c(p)±1, which never equal p's own color, so the two neighbor colors still need separate counts. Two counts of the same color would only certify two liberties if I can tell apart empty neighbors of that one color, and I'm not yet sure that's possible.

I'm trying a coloring where every neighbor of a point shares one color, so the empty neighbors all count as a single color, but that leaves an odd point with neighbors in two different colors, so I'm considering a four-color split instead.

I'm trying to find an encoding that avoids the two-count problem, so I'm checking whether computing the merged-group check at a later layer, from the neighbor information already held in the network, would sidestep it. The open issue is that the e-counts needed at that layer aren't cleanly available, since a threshold on the neighbor count isn't monotone.

I'm reconsidering the encoding so that the merged-group check no longer needs two separate neighbor counts. The idea is to split each empty point's neighbor count into two bands at the next layer, using ReLU units, so the needed information can feed the gating directly.

I'm checking whether a relu over the neighbor band at layer 5 could give a per-color count, but summing those inside the relu is nonlinear and doesn't work. So I'm looking at a different route: each empty neighbor's presence as a per-color indicator, OR'd across the neighbors of p.

I'm working out how to tell apart the empty neighbors of each point by color, so I'm defining a unit for each neighbor point that marks whether it is empty with the right color, and separately whether it is an own stone that contributes its group's colors except the one that might be the point itself.

I'm working out how to get the empty-point indicator for each neighbor by combining an empty-point check with an own-group presence check, so both cases can be handled by one unit per neighbor and color.

I'm now checking whether a single relu on a combination of the neighbor band, the own-group presence, and a constant could separate the three cases, and it can't, since requiring the empty case to be positive while the stone case stays nonnegative forces a contradiction. So I'll split this into two units per point rather than one.

I'm checking whether a two-unit split works here: a passthrough of the neighbor's band value and a ReLU of that band minus one, which together let the empty-point indicator be read off linearly at the next layer.

I'm checking whether keeping the b-band units at layer 5 is still needed once the O5 units are in place, since the layer is already at 567 units and that's over budget.

I'm checking how the own-group presence units fit into the layer budget, since the neighbor colors a point needs differ depending on its own color relative to the neighbor's.

I'm checking whether the opponent side can get its final atari indicator one layer earlier, by folding the NOT step into layer 4 so the threshold lands at layer 5. That would leave one unit per point at layer 5 for the opponent. For the own side, I'm laying out the timing: seed at layer 1, three propagation steps through layer 4, and the presence units at layer 5.

I'm checking whether the own-atari detection can be folded into the merged-group check at layer 6, since computing a separate NOT step and then a threshold would cost another layer I don't have.

I'm checking how to detect an own group in atari at the escape point p, since a per-neighbor test would cost too many units, so I'm considering whether a merged-safety bonus could be weighted by the count of such neighbors instead.

I'm weighing a combined value for each point, where the size of own groups in atari adds to the value and an unsafe merge subtracts from it. For the own side at layer 6, I'm planning units that flag whether the merged group at p would have fewer than two colors (one for each of c(p)±1), plus a separate check for own neighbor groups in atari.

I'm checking whether the own-side seeds fit in one layer, since certifying two same-color liberties doesn't reduce to a single relu, and at 243 units per side the first layer goes over budget.

I'm checking whether the seed for the opposite color can be made to carry a lower bound of two when two real liberties share a color, and whether that claim stays valid once the merged-group check reads it later.

I'm checking whether a claimed color set of size m guarantees m distinct liberties, so that a virtual color claim can still serve as a lower bound on liberties other than p.

I'm checking whether a virtual color claim can still act as a lower bound on the liberties other than p. If a stone's two same-colored liberties give it an extra color, the count of distinct colors other than c(p) should still bound how many real liberties remain once p is filled.

I'm checking whether a virtual color claim is still sound: a claimed set of size m should always mean at least m distinct real liberties in the relevant subset. Then I'm working out which virtual color to assign, using a seed built from the opposite-color count and a threshold on the two-liberty case.

I'm trying to get the layer-2 propagation under budget, since three propagation units per point on each side plus the board units would overshoot the 567-unit cap.

I'm checking whether two parity-based channels for the opponent side, one seeded on at least one liberty and one on at least two, fit alongside the own side's three mod-3 channels and the board units in layers 2 through 4. That leaves 26 spare units, but once the layer 5 pieces are added the count goes over the 567 cap again.

I'm sizing the layers for the opponent side: its channels go NOT at layer 4 and become the atari indicator at layer 5, with one unit per point. Layer 4 then fits at 486 units. Layer 5 is tight too, since the boundary and own-presence units plus the opponent atari units also land at 486.

I'm settling how an own group in atari should count toward an escape bonus at a point p, and checking whether that can be expressed without a separate per-neighbor own-atari unit. I'm also looking at whether the empty-point term, built from the board units with a small weight on the accumulator, stays close enough to 1 to work as a multiplier in the NOT units.

I'm working out how to detect whether an own neighbor group is in atari: the idea is to check whether the group's colors, seen from the empty point p, are all equal to c(p). I'm also tallying the number of colors other than c(p) present in the merged group at p, to see how many liberties the move would leave.

I'm working out how to detect an own neighbor group in atari from the empty point p: for each neighbor q, the colors other than c(p) must be absent, which I'll test with a NOT-style unit over q's own presence channels.

I'm working out how an own group in atari could feed an escape bonus. Per neighbor, the check costs two units at layer 6, and the total for own-atari detection is 162. The output stays linear: an escape term counts adjacent own stones in atari, and the unsafe penalty is a separate additive term, since multiplying the two would need another layer.

I'm settling on the escape term as a linear sum over adjacent own stones in atari, since a per-point product would need another layer. The unsafe-merge penalty stays as a separate additive term, so the output is just a weighted count of opponent atari neighbors plus own-atari neighbors.

I'm now checking how an unsafe merge interacts with an empty point that has no own neighbors: the merged group is just p and its empty neighbors, so the question is whether those empty neighbors carry both colors.

I'm working out the unsafe-merge check for an empty point p: it's unsafe when one of the two colors is missing from the merged group and there are at most one empty neighbors, so the condition is a NOT over the presence of each color plus a cap on the empty count.

I'm working out the unsafe-merge check as two terms, one for each color. Each term is a relu that fires when that color is missing from the merged group and there are at most one empty neighbors, and I'm checking that the double-counting case when both colors are absent still behaves correctly.

I'm working out how to penalize a self-atari move, and the merged group's size is unknown, so I'm weighing whether the number of own neighbors can stand in as a proxy for it.

I'm checking whether a unit that counts own neighbors can be folded into the unsafe-merge term so the self-atari penalty grows with group size. The worry is a single-stone invasion into opponent territory with one empty neighbor, which currently looks unsafe and could fall below pass.

I'm checking the unsafe-merge case for groups with several liberties: a neighbor group whose only liberties are the same color as p gives no other-color count, so the check has to handle an n_e of one on the other side.

I'm checking the unsafe-merge case where a neighboring own group has liberties that are all the same color as p, since that gives a false positive: the merged group actually has three liberties, but the check only sees one other color. I'm weighing how often that happens with mod-3 coloring on real positions, and I'm also looking at how opponent two-liberty detection for atari attacks can't distinguish two from three, so I'll keep that to local patterns. Then I'm tallying the budget at layer 6 for the opponent capture units and the own-atari checks.

Now I'm budgeting the last layer: the opponent capture units and own-atari checks come to 81 and 162 units, and the unsafe-merge units add another 162. The accumulator goes in at layer 6 as well, so the total lands at 486 with 26 spare, but I still need every new feature gated to zero at occupied points so the stones don't perturb it.

I'm working out the accumulator placement: putting it in the same channel as the board features would leak signal into occupied points, so I'm considering computing the local features late instead, from the layer-5 board and own-stone units.

I'm working through the budget for the last layer: the opponent-capture and own-atari units are already counted, so the question is what the output still needs from layer 6 and whether there's room left to compute local patterns there.

I'm now tallying the capture and escape units at layer 6: the opponent-atari count per point is 81 units, and the escape check needs 162 more, since each per-neighbor NOT-style test has to be its own nonlinearity and can't be folded into one sum. Those, plus the unsafe-merge units, are what I'm budgeting against the layer's capacity.

I'm working out how to get own-atari and merged-group liberty checks from the same NOT units, so the layer-5 features do double duty. Each own neighbor would contribute an absent-color flag, and the empty point's unsafe-merge test would be an AND over those flags.

I'm trying to avoid a bottleneck at the last layer by moving the opponent-related features earlier in the network.

I'm checking whether the capture bonus can be folded into the accumulator in b at layer 6, which would avoid a separate per-point sum. The concern is that at occupied points an adjacent atari stone could push b off its exact value of 2, which would leak signal into stones.

Since the last layer only feeds the output, which masks occupied points, I can fold the accumulator into b there freely. I'm planning to use the 81 units for the accumulator, 162 for the own-atari checks, and 162 for the unsafe-merge terms, which leaves 107 spare for local patterns. For the eye feature, I'll compute it at layer 6 from the layer-5 board and own-stone units, as a single unit per point.

I'm checking how the local features from layer 1 can ride along in b so they reach the output, gating them to zero at stones so the occupied points stay exact. At empty points the epsilon-scaled accumulator still stays under the gate threshold. I'm also noting that the empty-point term E_e in the layer-6 unsafe-merge units picks up that epsilon contribution, so I need to account for it there.

I'm checking the float32 precision of reading the accumulator back out through (b-1)/ε, and it works out to an absolute error around 6e-4, which is fine. The remaining issue is that the empty-point term O5 = relu(b-1) picks up ε·z at empty points, so I need to track that leak too.

I'm now weighing whether the eye feature and the remaining local patterns should sit in layer 1 or be pushed to layer 6, where about 107 units are still free after the accumulator, the own-atari checks and the unsafe-merge terms.

I'm weighing whether the opponent atari detection is worth its false positives: at radius 3 with par-2 it misflags about half the cases, and since the capture bonus is large, those false captures could outrank saving my own groups. I want real data to compare that against giving the own-side mod-3 features radius 4 instead.

I'm planning to simulate the approximate capture and merged-safety detectors on pass-1 boards, measuring their errors against the true values for each empty point, since both sides' quality matters.

I'm checking whether the opponent-channel trick could merge the b values with the opponent features, since b is zero at opponent stones and that might leave room to combine them.

I'm trying to merge the bias term with the opponent channel into one unit, but the opponent gate then kills those channels at opponent stones whenever the merged value is positive, so that doesn't work. Combining the bias with the own channel instead is what I'm checking next.

I'm checking the channel budget for each layer: with b plus five feature channels that's exactly 486 units, leaving 26 spare. The open choice is how to split those five channels between own-side and opponent-side features, either three own and two opponent or two own and three opponent.

I'm weighing whether to spend the spare channel budget on opponent-side features instead, but the own-side check is the harder problem: a single presence bit for "other liberties of parity 1-p" can only certify at least one liberty unless there are two or more empty neighbors, so I'm wondering whether seeding virtual claims from stones with two or more liberties would close that gap.

I'm going to test the capture and merged-safety detectors directly on real pass-1 boards, comparing a parity-2 coloring against a mod-3 coloring at radii 3 and 4 for both the opponent and my own groups. The seeding rule for the mod-3 virtual claims is still on my mind.

I'm checking how the opponent capture test and the own merged-safety test would work with the proposed seeding and propagation, judging an opponent group in atari when it has at most one colour, and judging a merged group safe when it sees two colours or two empty neighbours.

Own-side self-atari checks cannot be certified with parity-2 coloring: only one parity is ever available as "other liberty" from a neighbor's perspective, so the merged-safety test can only confirm at least one liberty unless there are two or more empty neighbors. That makes par2 weak for the own side, so the own side needs the mod-3 coloring with its three channels. That leaves two channels for the opponent side, which would mean parity-2 there, unless a different two-channel scheme does better.

I'm weighing whether the opponent side could use a different two-channel scheme than parity-2, since parity-2 gave 749 false positives at radius 3. A "has liberty" channel plus a "certified multiple liberties" channel is one option, though certifying a pair would need extra units at layer 1. I'm also setting aside the idea of shrinking the own side to radius 3, since that wouldn't free any width.

I'm weighing whether a unit could serve either color depending on the stone, since a plain ReLU output can't carry a sign and the weights are fixed, so that route looks blocked. I'm also considering whether units could be split by direction so each point only receives from certain neighbors, though that seems unlikely to help.

I'm trying to see whether the bias term can be folded into the own-side channel, but the own gate would then leak positive values into every own channel at that stone, so that doesn't hold. Dropping the bias another way, by letting each side's channels carry their own offset, is what I'm turning over now.

I'm considering a mod-3 coloring for the opponent side too, but restricted to just two of its three colors, so it fits in two channels. The idea is that an opponent group would be judged in atari when at most one of those two colors shows up among its liberties. I'm checking whether dropping the third color causes too many false positives.

I'm checking the opponent-side options now, including a parity-2 scheme with a pair certification. The budget at layer 1 is 405 units, leaving 107 spare. I'm also mapping out how the radius-3 and radius-4 steps would fall across layers for each side.

I'm now checking whether the opponent side can work with four propagation steps, which would need a 6-layer design, but the layer-5 budget comes out to 567 units, over the limit. I'm looking for a way to drop the own-count units at that layer so the budget fits.

I'm checking whether the own-side unit at empty points could carry an empty-point signal, since the same unit currently serves every point type and I need something that tells me a point is empty for the U± computation at layer 6.

I'm trying to see whether the empty-point information can ride on the own-side presence units at each empty neighbour instead of needing a separate channel, since the U± computation at layer 6 depends on it.

I'm checking whether an empty point's own channel can signal that it is empty and belongs to color k, so that the merged-safety check at the next layer picks up empty neighbors automatically. The difficulty is that a single ReLU unit at L5 has to behave differently for empty, own and opponent points, and the bias-based gating I'm considering doesn't separate those three cases cleanly.

I'm checking whether the empty-point signal can come from the b unit at layer 5 by using a different function than the current gating, since the three point types need to be told apart and the bias-based approach doesn't separate them cleanly.

I'm checking whether the own-side empty-point information could be computed one layer earlier, at layer 5, directly from the layer-4 quantities, so the liberty-count units at layer 6 wouldn't need it passed along separately.

I'm checking whether the own-side empty-point information can come from the L4 quantities, since routing the path through neighbours doesn't work when the middle stone might be empty or opponent. The other route is to compute an O5 indicator at layer 5 from b, which costs 81 units, and then recover the empty-point signal at layer 6, but that puts b and O5 together at 162 units.

I'm checking whether reducing the own side to two colours per point would work, but the stone-side reads at layer 6 need all three colours of each stone, since both the empty-neighbour checks and the own-atari checks use the colours other than the stone's own. So that reduction doesn't fit, and I'm moving on to the option of cutting the opponent side to two colours with radius 3.

I'm checking whether opp R=3 with parity-2 holds up, since that's the weakest spot with 749 false positives on the heuristic data, and I want to see how it actually does on real positions.

I'm checking that par2 already includes the pair certification, since adjacent stones take different parities. Next I'll run the evaluation on real positions from pass_1 for the side to move to see how often it errs.

