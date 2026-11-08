I'm checking whether stone (1,2) is really in atari, and it isn't: its link to the certified stones runs the long way round through the top-left corner. The diagonal to (2,1) doesn't connect directly either, since (1,1) is empty and (2,2) is an opponent stone, so I'm tracing the other path around.

The false atari flag comes from a path that runs four steps around the top-left corner, and the propagation only reaches three. Because of that, the group looks weaker than it is, and the eye-suppression, escape bonus and safety count all follow from it. So the fix has to deal with the propagation radius rather than the single stone.

The fix probably needs a longer propagation reach or better local certificates. The stones next to (1,2) in its string have zero liberties of their own, so the only breathing room sits at (1,1).

I'm now looking at a diagonal-pair certificate: two stones that are diagonal to each other and share corners that are either own stones or the same empty point. For (1,2) and (0,1), one corner is (0,2), which is an own stone, so they are already in the same string, and this idea doesn't add anything new.

I'm weighing whether a local certificate could show that (1,2)'s string has two or more liberties, but its liberties at (1,1) and (3,0) are far apart. Any such proof would have to propagate through the string, which needs four steps where only three are available. For an eye point like (1,1), I'm considering whether filling it is bad whatever the atari flags say, unless the neighboring stones belong to different strings.

I'm weighing whether to keep the eye penalty unsuppressed when the adjacent stones are flagged. Here the real case is a string in atari that could be saved by connecting through the same point, so the penalty would outweigh the escape bonus and block that save. I'm trying to judge which mistake is more common: filling a real eye because of a false flag, or missing a rare connecting save.

I'm weighing a rule that suppresses the eye penalty only when an adjacent own stone is flagged atari and another adjacent own stone is certified. The trouble is that mixed flags could mean either a real connecting save or a false flag inside one string, and I can't tell those apart from the local picture.

I'm weighing whether the eye penalty should be suppressed when an adjacent own stone is flagged atari, since mixed flags could mean either a real connecting save or a false flag inside one string. I'm considering better flags, possibly by extending the propagation reach for own strings beyond the current three steps.

I'm considering whether the eye penalty could be restructured so it doesn't need a further propagation layer, perhaps by letting an adjacent neighbor's propagated atari state feed in linearly to the output.

I'm considering swapping the eye suppression for an additive escape bonus driven by a one-step-deeper flag, so that a false atari at radius 3 gets corrected before it affects the eye penalty. That would mean computing the deeper flags for the neighbouring stones, which costs extra units in the next layer and conflicts with how much room the existing layers already use.

I'm working through the layer budget now: the NS6 and NO6 units would add 162 units to the L6 sizing, pushing the total to 648, which is too many. My tentative cut is the Terr unit, though I'm not sure yet whether dropping Iso would be the better trade.

I'm checking whether one shared unit per point can serve both the escape and capture flags, since the two supports never overlap: NO6 sits on opponent stones and NS6 on my own. That would save the extra layer budget I'm short on.

I'm checking whether a single shared unit per point could cover both the escape and capture flags, since the two supports don't overlap. Trying to fold both relus into one unit fails, because an opponent neighbor's certificate would wrongly certify an own stone, so I'm ruling that out.

I'm considering whether the output could sum escape and capture terms directly over neighbouring stones, and I'm weighing whether reusing the layer-one spare units could start own-group propagation earlier, though the layer-two room looks too tight for that.

I'm working out how to cut the layer budget: dropping the Terr unit and keeping the eye and propagation pieces in layer six, since a single shared unit can't serve both the escape and capture flags.

I'm checking whether dropping the unsuppressed escape-count layer frees enough budget, since the L6 units are already near capacity. Dropping the opponent ≥3-class unit would save room, but it's only used for the attack bonus, so I'm weighing whether a cheaper proxy could stand in for it.

I'm considering dropping the 26 unused units elsewhere: if the upper-level attack term and its propagation go, the freed space in the earlier layers could host the territory unit at layer five instead of six.

I'm weighing whether dropping the unsuppressed attack term frees enough room in layer five to hold the territory unit, though keeping the iso unit in layer six would still overflow the budget. Dropping the third attack-count unit U3 might be the cut that makes everything fit, since it's the one meant to handle single self-atari stones.

I'm weighing dropping the Iso unit and the UO attack term to free up layer budget. Iso is removed to avoid the layer overflow, and I'm starting to lay out the layer-by-layer unit counts, beginning with layer one, where I have 188 spare units.

I'm weighing whether dropping Terr and folding the eye and propagation pieces into layer six keeps everything within budget, and I'm checking the leftover room in layers two through four.

I'm settling the escape and capture terms: the escape bonus uses the NS6 flags on neighbouring own stones without any suppression, and the capture term sums NO6 over the neighbours. The eye penalty then drops its suppression step and just applies a -15 penalty with a +10 escape bonus.

I'm weighing eye penalty at -12 against an escape bonus of 12, so a real connection-save through an eye ends up near zero and competes with positional terms. I'm still uneasy about false flags, though I'm relying on the radius-4 NS6 flags to be mostly accurate. Next I'm planning to restructure the build into a version 3 with these parameters.

I'm settling on the final layout with the escape and capture terms read straight from the neighbouring NS6 and NO6 units, and I'm planning to write the third version of the build from scratch, compactly.

