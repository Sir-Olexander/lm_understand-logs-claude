I'm weighing a fix for the repetition leak, which is worth about +4.7% and the biggest known one. The question is whether the Iso functionality can fit into L5 using NO4's radius-2 flags, since both L5 and L4 have 23 spare slots.

I'm considering dropping the UO propagation at L5, which would free 81 units but lose the attack feature. Iso could then be computed at L5 using NO4 and folded into Pos as a negative linear input, provided Pos's offset keeps it non-negative.

I'm checking whether the Iso work could go into L5 if I drop UO there. That would free 81 units, but the attack feature goes with it. I could keep attack by computing it as its own L5 unit, but that uses the same 81 units, so it doesn't help Iso fit.

I'm now weighing whether to replace the UO propagation at L5 with the Iso computation, since both layers have spare slots. Keeping attack alongside Iso doesn't fit, because a separate attack unit would need the same 81 units. So the choice is between losing attack and losing the Iso fix.

I'm weighing a test that swaps the UO propagation at L5 for an Iso unit computed from the L4 units, folding it into Pos as a negative input. The trade-off is dropping the attack feature, whose value is unknown, against the roughly +4% expected from fixing repetition. Since I have passes to spare, I'll run one test to decide.

I'm dropping the attack feature for this test, and setting the L5 weight on the no-liberty term to its cap, so the Iso unit can be built as an option called 'iso5' in build4.

