I'm checking the capture probabilities: every iso capture comes out near certain, yet the pass estimates are essentially zero. That suggests either the estimate was low or the capture logit is very large. I'm working through the logit terms: twelve per adjacent atari stone, then a penalty from the liberty count, and a positional term from the contact count.

I'm working on a stronger penalty for isolated captures, since they carry a real repetition risk. I'm weighing whether to fold that penalty into the NS6 unit at each point, gated by whether the point is empty or occupied.

I'm checking whether a combined unit per point, gated by whether the point is empty or occupied, can carry both the own-stone term and the isolated-capture penalty. The capture term only switches on at empty points, and the other term then just subtracts the count of certified neighbours, which is zero when the point is isolated.

I'm checking whether a single unit per point can carry both the own-stone term and the isolated-capture penalty, but a single relu can't cleanly switch between the two. The empty-point term is already non-positive when the point is occupied, so the trouble is the occupied-point term, which can go positive for an own stone surrounded by opponent stones.

I'm checking whether a single unit per point can switch between the own-stone term and the isolated-capture penalty by subtracting a large constant from the capture part when the point is occupied, which is enough because that part is bounded above.

I've ruled out folding the isolated-capture penalty into the same unit as the own-stone term, since the gating can't be made exact with a single relu. So I'll need separate units for it, and I have 20 spare in that layer. As an alternative, I'm considering lowering the bonus for single-stone captures in general and raising it for multi-stone captures.

I'm considering whether a single unit can handle both the own-stone term and the isolated-capture penalty, but the gating doesn't work out cleanly. Another option is to make the M penalty stronger for isolated captures through the T coefficient, though the difference in A between an isolated capture and a normal contact move is only 0.25, so it's hard to separate them that way.

I'm testing a softer scaling for the severe-penalty term: with the single-capture weight at 0.1, an isolated capture would only pick up a small penalty. But with the weight at 0, a capture filling the last eye would get a huge negative value and never be played, so that setting breaks big captures.

I'm testing whether a penalty can single out isolated captures, since a scaling weight of 0.6 leaves the eye-capture cases unpenalized, but also fails to penalize the isolated single-stone capture, which has an A value of 1.6 and so zeroes out.

I'm ruling out a single unit for both jobs, since the gating can't be made exact, so the isolated-capture penalty needs its own units. I'd have to find about 60 spare units in that layer, and I'm considering whether to drop something or move the feature into an earlier layer computed from the previous one.

I'm considering merging the Terr term into the Eye unit as a single territory score, with a heavy penalty on points whose neighbours are all friendly.

I'm working out whether a territory score can absorb the eye penalty: an interior eye should score around 4 to 6 and get a penalty near 12, an edge eye a smaller one, and an ordinary territory point only a small penalty.

I'm settling on a merged territory score: an interior eye should land around 5, with a corner eye adding a bit more penalty and an edge eye a bit less. That would free up 81 units, enough to hold the isolated-capture penalty in its own unit, and the whole layer would fit exactly at 492.

I'm settling the design: the Eye and Terr units merge into one territory score, and a separate isolated-capture penalty gets its own unit, with the capture logit then landing around 2 to 4 so passing wins when ahead.

