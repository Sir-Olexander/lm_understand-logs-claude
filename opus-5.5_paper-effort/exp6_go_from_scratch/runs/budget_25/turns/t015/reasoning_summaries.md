I'm comparing par2 and mod3 on real data, and par2 comes out ahead. Its virtual certificate for stones with at least two liberties covers every such stone, and I'd expected mod3 to do the same, so I'm checking why it doesn't.

Par2 has a pair certificate for free, since adjacent stones have opposite parity and their liberties land in different colors, but mod3 can't guarantee that because neighboring stones' liberty colors may coincide. Even so, false captures remain frequent for par2 at radius 4 (1617 against 3588 true), so I want to see how much of that comes from the radius limit versus color collisions by checking the unbounded case.

I'm working out whether a radius-2 signal can be built from radius-1 pieces: for par2, a stone's own liberties give one color, and a neighbor's liberties give the other, so I'm looking at whether an OR of ANDs across neighbors can be approximated with a single threshold instead of two layers.

I'm considering whether a larger radius could come from linking stones along straight runs, where a stone two steps away along a line is connected through the middle stone, though a general doubling scheme would be too expensive.

I'm weighing whether a seed can fold in the neighbor's liberties directly, since the path-wise AND works in one unit but summing across several paths to get an OR might not hold up.

I'm working through whether each seed unit can be an AND over a neighbor's color, by summing the empty liberties of the other neighbors, so that a single threshold captures the OR inside it.

I'm checking whether a three-way weighting can make the threshold sum fire only when both neighbor terms are on and at least one empty liberty is present, since a plain sum lets one term plus two liberties pass by mistake.

I'm checking whether a per-edge unit that requires both stones and an adjacent lib of the right color could carry the one-step propagation, but at 288 units per side that's too costly for the first layer. So I'm falling back to accepting the radius limit and considering whether R=5 is worth trying.

I'm checking whether the final output layer can read the atari signal directly: after one more step, the count of absent colors gets summed across both colors, and for a stone next to an empty point at least one color is always present, so the sum itself would mark atari without needing another threshold.

I'm checking whether the output can work with six steps instead of five: the last layer could use NOT units gated by the opponent mask, and then the sum of the two color counts would directly flag atari without another threshold.

I'm checking whether the own side can also use par2 with five steps for atari detection, which leaves the self-atari case still unsolved.

I'm now checking whether the own-side self-atari test can reuse the opponent pipeline at the empty point: the merged group's other liberties come from empty neighbors and from adjacent own groups, and with par2 only the opposite color is reliably countable, so a plain "two liberties" count doesn't carry over directly.

I'm now looking at whether self-atari of a larger group can be caught by counting, for each neighboring own group, how many liberties it has besides the point being played, and I want to see how often self-ataris happen when the group already sits at two liberties.

I'm going to count how many distinct cases there are for the saveable own groups, since many of them may stay in atari across several moves.

