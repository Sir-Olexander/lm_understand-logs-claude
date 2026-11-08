I'm checking how often the true and false positive cases fall at each density: true atari stones mostly sit at density 1–4, while false positives cluster at 4–8. High-density cases are costly to veto outright, since the veto at density 6 or above would drop many real captures along with the false ones, so I'm considering a softer penalty that shrinks as density rises.

I'm weighing a soft penalty that fades with density, but it would still discount true captures of big groups in dense areas, which are the most valuable ones. Vetoing only the self-atari suppression instead might keep those captures, though for dense true captures the penalty could still outweigh the bonus, so the trade-off remains unclear.

I'm weighing whether to drop the density veto for the opponent side and rely on a simpler proxy, since the layer budget is already used up. The only candidate I can see is counting opponent stones with liberties nearby, but that count isn't normalized, so it doesn't work as a clean density measure.

I'm working out whether the opponent-side density signal can share a unit with the own-side attack count at layer 6, since a point is only ever one or the other. The idea is a combined unit that fires for an opponent stone when its four-neighbour count is high relative to its neighbourhood, and for an own stone when it touches multiple of my groups.

I'm checking whether a combined unit can serve both sides: an opponent stone's neighbour count and an own point's attack count. The obstacle is that an own point gets pulled down by the opponent-channel terms meant for the opponent side, so the shared unit would suppress the very attack signal it is supposed to carry.

I'm checking whether one shared unit can serve both sides: the at-count for my own points and the neighbour count for opponent stones. Folding the opponent-channel suppression into that unit would wrongly cancel my own attack signal, and a static weight can't tell whether a neighbour is own or opponent.

I'm checking whether the opponent-side neighbour count can share one unit with the own-side attack count, using equal weights for escape and capture so the output can read it without knowing the colour. The problem is that an own point's attack count gets cancelled by the channel terms from opponent neighbours, which are positive in that case, so the shared unit would suppress the very signal it is meant to carry.

I'm testing whether the eye-detection term can share a unit with the opponent-side neighbour count, since empty and opponent points never coincide. The catch is that an opponent stone surrounded by my stones would make the eye term fire alongside the neighbour term, which would corrupt the signal.

I'm checking whether the opponent-side eye term can be merged with the neighbour-count unit, since an empty point and an opponent stone never coincide. At an opponent stone the eye part stays at or below zero, because it always has a non-own neighbour, so it can't corrupt the neighbour signal there. At an empty point the neighbour term pushes the sum down, which seems to keep the two sides separate.

I'm trying to fold the eye-detection term into the same unit as the opponent-side neighbour count, scaling the eye part by two so an eye at an empty point comes out at one while the neighbour term stays at or below zero there. At an opponent stone the eye part can't go above zero, since such a stone always keeps a non-own neighbour, so the neighbour signal stays clean on that side.

I'm weighing whether the opponent-side eye term can be folded into the same unit as the neighbour count, but the sign of the eye part goes negative at some opponent stones and would suppress the neighbour signal there. Positive-only scaling doesn't fix that, so I'm considering whether to drop the merge and rebalance the layer budget another way instead.

I'm trying to see whether a single relu can express the value function, since two linear functionals would need two units. Splitting it into a positive part and a negative part of the difference doesn't work, because the output can only read the opponent-side term linearly if it lives in its own unit.

I'm considering whether the value function can be expressed through fewer units: the own-side attack counts and opponent-side neighbour counts might be cheaper to compute at layer 4 with a smaller radius, which would free budget in layer 5 for the carrier that the linear read-out needs.

I'm now considering whether to move the own-side m-units down to layer 4, which would leave room in layer 5 for density handling, though that still costs budget in layer 6. Layer 6 wouldn't gain anything from it, so the trade-off comes down to how much density-veto suppression is worth keeping. Since self-ataris cost only about one per game against roughly 2.2 losses from the lost games, that suppression may not be worth the layer budget.

I'm considering a shortcut for the own-side pressure: a single term in the combined unit that approximates the radius-5 information, which would only be penalised when an opponent neighbour is itself in atari-like shape. The worry is that the penalty would also fire in ordinary positions where none of those neighbours is near capture, which would swamp the signal.

I'm concluding that the R5 suppression won't help either, since the false positives I found at R5 also had a neighbour count of 1. The density or size veto, applied to both the bonus and the suppression, is the more likely fix.

I'm considering whether the density penalty could go inside the neighbour-count unit at layer 5, using the net influence around each opponent group, since own and opponent counts can be read linearly from the layer-4 units. The worry is that large opponent groups would be penalised even when the capture signal is real, so I'm checking whether that net count separates the false positives from the true cases.

I'm considering whether the density veto could live inside the neighbour count itself, using the net own-minus-opponent count around each point. True atari stones tend to have many own neighbours and few opponents, so that measure might separate them from the false positives, but scaling it into a clean binary signal is a problem.

I'm trying a density penalty built into the layer-5 neighbour count, based on the net count of my own stones minus opponent stones in a radius-2 window around each point. True atari stones tend to sit among my stones, so that net value might separate them from the false positives. The open question is how to turn it into a clean signal without it getting too large.

I'm now checking the net own-minus-opponent count within radius 2 of each candidate, and the idea is to set a threshold so the density penalty only bites when opponent stones dominate the window. Before deciding on the scale factor, I want to measure that quantity separately for the true atari cases and the false positives.

