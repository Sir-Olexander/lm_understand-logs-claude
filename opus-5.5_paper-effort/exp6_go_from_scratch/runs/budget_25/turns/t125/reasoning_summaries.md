I'm checking the line-1 captures, where many moves have a positive capture count. A single corner stone captured by a point with no friendly or empty neighbors comes out well ahead without suppression, so that suppression is what matters here.

I'm settling on a choice now, since every variant I've tried lands around 75%. The S2R1 pattern I was looking at, a single stone on line 1 with two liberties and an adjacent opposing stone in atari, is shared by many true captures, so I can't separate the good cases from the bad by looking locally. I'll accept that and use the remaining passes to test one or two low-risk tweaks before confirming.

Next I'm weighing whether color-specific tuning could help: higher fighting weight as black, since the as-black results trail white. The territorial variant did badly for both colors, so the fighting weight is the lever to test, not territory.

I'm working out the cost of a 2-liberty atari attack on an opposing single stone: a per-point score built from neighbor counts, with a large penalty for adjacent opposing stones. That only fits about 22 spare units, so most points would be left out, and I'm considering whether the same term could ride on the existing per-point units instead.

I'm working out whether the captures on line 1 can be separated from the eye-filling ones by a local rule, and it looks like they can't, since the same pattern shows up in both true and false captures. I'm now checking whether a combined per-point term could cover the atari case for opposing single stones within the remaining budget.

I'm testing a combined per-point term for opposing single stones, folding the atari bonus into the existing line-1 unit so it fits the remaining budget. Two-liberty singles would land near zero and atari singles would score higher, which is the separation I'm after.

I'm dropping the local atari term, since folding it into the line-1 units would also suppress the multi-stone cases. Several variants are sitting at about 75%, so I'll compare v13, v15 and v16 over more runs before choosing one to keep.

I'm settling on v16 as the final variant, since v9, v13, v15 and v16 all sit at about 75% and v16's edge certainty fix is a correctness gain. The remaining few passes could go to a black-specific opening tweak or a check against the weak and random opponents for the record.

I'm dropping the idea of suppressing the bonus by a high walk-count threshold, since that would also strip value from moves that are actually good. Restricting the veto to moves with an own neighbor doesn't work either, because the per-point term has no way to see which move is being played. I'll set this line of thinking aside and look at the opening instead, starting with the first moves I played as black across the 20 games.

