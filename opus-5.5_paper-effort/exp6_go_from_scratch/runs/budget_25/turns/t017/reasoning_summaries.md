I'm checking the false-positive counts for the cap condition per legal point, and they drop as the radius grows, from 756 at R=3 down to 392 at R=5 against a target of 2506. I'm also noting that the unsafe total is 9279, which includes many single-stone self-ataris in opponent territory.

I'm working out why the escape false positives are so high, about 1015 at R=4 against 1372 for the tesc total. The likely cause is that two odd stones in the same group sit at even distance, so the P3 pair certificate can't tie them together and a group can be wrongly judged in atari, much like the par2 collisions. I'm also weighing whether combining the class and merge methods, so a move counts as safe if either says so, would cut the merge method's extra false positives.

I'm weighing whether the same parity mismatch behind the escape false positives also hits atari detection for own groups, since that's the same problem as the cap check. Two odd stones in one group each showing a single liberty of the same color would be judged in atari wrongly, while pairs of different-parity stones can still be tied together.

I'm comparing par2, P3, and mod3 on the same atari tests, since P3 looks like it should do at least as well as par2 for own-group atari detection. Then I'm wondering whether the radius is the main driver, so I want the false-positive floor at large R. The real question is how much these false positives change the chosen move, not just the raw counts.

I'm weighing whether to drop the merge approach and use the class method with per-group classes computed at a deeper level, since the simpler scheme isn't clearly worth its budget. The class method's 104 false negatives out of 9279 unsafe moves look acceptable, and the false positives on the escape and capture checks are mostly harmless, so I'm leaning toward keeping it simple.

I'm now sketching the class-based check: a move is unsafe when the empty-neighbor count plus the per-group class count is at most one, with the per-group counts coming from the own-group NOT units.

I'm checking whether the X2 indicator should be 2 minus the absent count, since with that form a neighbor with zero absent points lights up at 2 and one absent point at 1.

I'm checking whether the X3 term can go negative when a group is in atari, since a negative value there would push the unsafe count up even when another group has three or more liberties, and I want to keep that from happening.

I'm now weighing a way to keep per-group counts linear: at the next level, the "at least three present" and "at least two present" conditions would need their own threshold units, since a plain linear count can go negative for a group in atari and push the unsafe count up wrongly. I'm considering whether to propagate capped counts instead, which would avoid that sign problem.

I'm now looking at how to keep the propagation signals non-negative, alternating between positive and NOT polarity from one layer to the next, so the AND and OR steps stay linear.

I'm settling on propagation signals that alternate polarity between layers, so the NOT layers hold exact binary values and the AND and OR steps stay linear. For the gating, points outside the current set have to read as absent so they don't disturb the AND, and I'm checking how that interacts with the neighboring points that are in the set.

I'm checking whether the gating can be done with a single unit at each non-set point, since a neighbor-dependent gate would throw off the AND and OR steps.

I'm settling on a per-stone deficit count for the own-side check, so each stone's missing colors can feed the cap condition.

I'm considering whether to drop own-side propagation to radius 3 with three colors and keep the opponent side at par2 with radius 5, since the nonlinear max-over-neighbors step doesn't fit a purely linear layer.

I'm weighing whether to keep the bias unit b as a separate signal through the middle layers, or to fold its gating into the channels directly. I'm also considering using three channels on the own side and two on the opponent side, both at radius 4, and finding the spare capacity elsewhere.

I'm laying out the layer-5 units for the opponent side: the 162 NOT units gated so that non-opponent points read as zero, leaving the atari count per stone in the output, and I'm checking whether the layer-6 readout can carry those through or should fold them into the capture bonus.

I'm working through how the opponent-side atari counts fold into the capture bonus at the output layer, and I'm checking whether the own-side features can still get the empty-neighbor information they need without an extra layer.

I'm settling on the class-based unsafe check at layer 6: the empty-neighbor count can ride along in the accumulator, and two linear indicator terms per own stone would gate the two threshold units.

I'm still trying to make the own-side class counts come out of a linear layer, since a "at least two colors present" test isn't linear on its own. A plain sum would only tell me that some color is present, so I'm considering whether the layer-4 own units should already encode the class thresholds.

I'm checking whether own-side class counts can come out of a linear NOT layer by computing a per-stone deficit at layer 4, so the layer-5 thresholds for "at least two colors" and "at least three colors" stay nonnegative. That would keep the own side at radius 3 effectively, and the unsafe check at layer 6 could then use those class indicators.

I'm settling the layer-5 own-side units now: class indicators c2, c3 and an atari indicator per own point, three signals per point and 243 units in all. With the 162 opponent units, layer 5 comes to 486 units total, which fits the budget.

I'm weighing whether the escape bonus can read the at-liberty signal at layer 6, which would keep the own side at radius 4, while the self-atari check has to use the class counts from layer 5, so the own side would effectively stay at radius 3 for that part.

I'm weighing whether the opponent side could take radius 3 instead, so that the atari count comes straight out of the NOT layer at L4 without spilling into L5.

I'm checking whether the capture bonus folded into b5 would leak into the own-stone gating: an own stone next to an opponent atari point would get a small nonzero b, which would break the gate on the own class units unless that leak can be neutralized.

I'm checking whether the leak from the capture bonus into the gating signal stays small enough to avoid pushing the class units off zero, since a few hundredths of error multiplied by the large unsafe weight could still tip the result.

I'm now seeing that the class-based unsafe check at layer 6 can't use neighbor class counts at all, because those counts only exist one layer earlier, so I'd need to push either the own radius or the class computation somewhere else.

I'm comparing a few more options in simulation: own-side radius 3 with class counts, own-side radius 4 with a merged-color check, and the atari indicators at radius 3 or 4, all paired with opponent-side par2 at radius 3 to 5.

I'm now reconsidering a merge-based check where p is treated as a virtual stone, so the colors of its own local liberties get counted alongside those of its neighboring groups. Once p is filled, it stops being a liberty, so I'd drop its color from those neighbors' claims to stay conservative, and I'm checking whether the combined claims reach two colors under the P3 scheme.

I'm checking whether a merged-group check works: treating p as a virtual stone, the colors claimed by its neighboring groups, minus p's own color, plus the real colors of its empty neighbors, need to reach two. That gives 4137 false positives at R=4 under the P3 scheme.

I'm checking the false-positive rate for the combined unsafe test, comparing the class-based and merge-based checks across several radius and coloring options. I'm running that comparison for both the own and opponent sides so I can see which combination actually cuts the false positives.

