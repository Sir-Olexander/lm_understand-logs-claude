I've confirmed the new seeds are exact, so no bad seeds get through. The union A (F∪F3) stays the same, since the new F already covers what the old F3 did. The 2-library groups are now exact, with all 65239 correctly counted as at most two. But 44908 of the 539916 groups with three or more libraries are falsely flagged as T, about 8.3%, compared with 1.4% under the overcounting version. The old approximate F3 seeds did catch many real three-or-more groups, which is what I'm weighing against that cost.

I'm weighing the cost of each error type: calling 8.3% of the ≥3-library groups "≤2" versus missing about 11,257 two-library groups. Marking large groups as having few liberties would wrongly discourage some necessary connections, while the old approximation caused self-atari blunders. I'm looking for a middle path that keeps the exact seeds and adds reach without overcounting.

I'm now going to test the new seeds against the old ones on decision-relevant cases: empty points with at most one liberty next to my own stones. For each one I'll compare the true self-atari status after the move with what the current and new seed rules each predict.

I'm checking the merged group's liberties after captures, to see whether the move leaves it with one liberty or fewer.

