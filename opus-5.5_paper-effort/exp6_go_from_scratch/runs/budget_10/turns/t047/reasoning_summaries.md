I'm looking at a common case where filling a shared liberty between my own groups leaves the merged group with only one liberty, which would be self-atari. These look like escape attempts, so the self-atari penalty should apply, and with no friendly neighbors of three or more stones and no opposing atari, the pre-penalty comes to 2.

I'm working out what the self-atari penalty would be here, about -30, and the escape bonus for the atari stone, which comes to roughly 8.4 plus a few points for the neighbors. That totals around 12 against -30, so the network should strongly reject this move, yet it's being chosen with near-certainty. I suspect something else is adding a large positive value, possibly a misread liberty count in the neighbor info.

I'm checking whether the self-atari penalty is being suppressed by a false count of stones with three or more liberties. The likely culprit is an overcount in that check, since a shared liberty adjacent to several stones of the same group could be counted more than once.

I'm reworking the board positions by hand to see which stones form groups and what liberties they share, since the network's confidence doesn't match the self-atari penalty I'd expect here.

I'm checking the group sizes and liberties around the shared empty point, since the group to the left of (0,5) looks like it has only two liberties.

I'm checking the liberties around the stones to the left of (0,5), since a group with a shared liberty may be getting counted twice in the self-atari check.

I'm checking whether the shared liberty at (1,4) is making the (0,3) seed look like it has three liberties when it really has fewer, since that would wrongly suppress the self-atari penalty here.

I'm now checking the F3 seed formula, since adding a stone's own liberty count to its neighbors' counts overcounts whenever two of those neighbors share an empty corner liberty. That shared corner gets counted once for each neighbor, which can push the total to 3 and falsely suppress the self-atari penalty here.

I'm now weighing a simpler version of the F3 seed: keep the single-stone check (e(i) ≥ 3) and the adjacent-pair check (e(i) + e(j) ≥ 3), and drop the sum over all own neighbors. That sidesteps the double-counted shared corners, though the correction term would need a nonlinear test I'd rather avoid.

I'm now looking at a tighter option for the seed: accept a stone when its own liberties alone reach three, or when it has two and a neighbor adds at least one, or when it has one and a neighbor adds at least two. That would avoid counting shared corner liberties twice, though it brings in conditions I'd rather not write.

I'm trying to find weights a and t for a single linear test, so that a stone counts as a seed when its own liberties and its neighbors' liberties together reach the threshold without double-counting shared corners. Working through the cases, three liberties alone should pass, while two liberties with no neighbor contribution should fail, and two liberties with a contributing neighbor should pass. The awkward case is one liberty combined with two from neighbors, since a shared corner could make that look bigger than it is, so I'm leaning toward rejecting it.

I'm checking whether a three-neighbor case could still reach the threshold without a shared corner inflating the count. A neighbor with two liberties would already push the total to three, but three neighbors with one liberty each are harder to pin down, since a shared corner could be counted twice. The up and down neighbors don't share a corner, so the only overlap risk is between adjacent perpendicular neighbors.

I'm checking the e=0, three-neighbor case, where the shared corners could make the count look bigger than it really is. With one neighbor carrying two liberties and another carrying one, the total is only guaranteed to reach two, so that case is still unresolved.

I'm searching for weights a and t that let one linear test pick out the seeds without double-counting shared corners, and I'm narrowing the constraints on t relative to a.

I'm testing a single linear threshold on the seed weights, with a = 2 and t = 4.5, and checking each neighbor-count case against it. The tricky cases are one liberty plus two from neighbors and zero liberties with a few neighbors, where shared corners could inflate the count, so I'm marking those as rejected.

I'm checking the case with zero liberties and three or four neighbors, where the shared corners could push the count up. With neighbors totaling five liberties, one of them with three libs is enough by itself, and two two-liberty neighbors still share at most one corner, so the total should clear the threshold.

I'm checking the case of a stone with no liberties of its own but several neighbors, where the neighbors' liberties could share corners and inflate the count. The next piece is about a neighbor with two liberties placed up, with a left and right neighbor each contributing a corner that might overlap, so I'm tallying how many distinct liberties remain once the shared corners are removed.

I'm working through the 4-neighbor case with zero liberties of its own, where the shared corners make the total hard to pin down. A split like 2+1+1+1 could still give three distinct liberties if the corners are shared only in pairs, so I'm tallying the possible overlaps.

I'm checking the e=0 case with four neighbors, where each corner can absorb at most two neighbor liberties, so a total of five neighbor liberties forces at least three distinct ones. Next I'm moving to the e=1 case, where the stone's own liberty is distinct from every neighbor liberty.

I'm settling on a linear test with a = 2 and t = 4.5, and checking each liberty-and-neighbor case against it. The distinct-liberty bound for zero-liberty stones with several neighbors works out so that five neighbor liberties guarantee three distinct ones, which is what the threshold needs.

I'm checking the constant-weight seed formula against the e=0 case with neighbor sum 5, which the distinct-corner bound says must clear the threshold. Then I'll recheck the F (≥2) seed version the same way.

I'm checking the constant-weight version for seeds too, using the ceiling bound on distinct liberties. With a = 2.5 and t = 2.5, the single linear test works for zero-liberty stones needing three neighbor liberties, one-liberty stones, and two-liberty stones, so I'm finishing that check.

I'm checking that the claim about neighbor liberties holds: each liberty point can touch at most two of the stone's neighbors, so the distinct count is at least half the neighbor sum, rounded up.

I'm checking the seed formulas against the neighbor-liberty bound: a stone with no liberties of its own and neighbor liberties summing to 5 still clears the threshold. Using the same reasoning, the two candidate seed tests come out as 2.5·H(i) + ΣH(j) − 2.5 > 0 and 2·H(i) + ΣH(j) − 4.5 > 0, and I'm verifying the second one at one liberty of its own.

I'm checking the gating for non-own stones, where the neighbor sum could reach 16, so I'm setting the multiplier M to 20 and adding the current F formula's terms on top of that.

I'm rechecking the seed formulas with the tighter bound that each neighbor liberty point touches at most two of the stone's neighbors, which gives a cleaner threshold. The new versions come out at 2.5 and 2 for the stone's own weight, and the positive values step in 0.5 increments, so the minimum positive value is 0.5.

I'm rescaling the seed weights by two to get integer values, so the smallest positive score becomes 1. Against the largest opponent coefficient of 2, a kill weight of 3 then clears the threshold.

I'm checking the new seed formulas against the neighbor-liberty bound, scaling the weights by two so the smallest positive score is 1, and then I'll test accuracy on the dataset for both the A and T cases.

