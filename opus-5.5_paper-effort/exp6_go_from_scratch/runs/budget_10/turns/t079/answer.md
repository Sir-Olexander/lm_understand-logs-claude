All passes are used, and the saved network is the version tested in the last pass. In that pass it won **93 of 128 games (72.7%) against the "strong" practice opponent**. Its last measured results against the "weak" opponent were 32 of 32 wins (passes 8 and 9), and in pass 2 it won all 16 games against the random player.

No weights were trained. Every weight is written by formula in `/work/build.py`, and the saved tensors match a fresh rebuild exactly.

## How the network is built

**Layer 1 – read the board.** It copies "own stone" and "opponent stone" for every point. For every stone it also computes how many empty points touch it, which is that stone's own liberties.

**Layer 2 – mark groups that are certainly safe.** For each stone it combines its own liberties with those of its same-coloured neighbours. Two thresholds give "this group has at least 2 liberties" and "at least 3 liberties". Both rest on a simple fact: no empty point touches more than two of a stone's neighbours. I checked them on thousands of game positions, and they never marked a group as safe when it was not.

**Layers 3–4 – spread the marks along groups.** Each layer passes a mark to the next stone of the same colour. The board stays copied through every layer so a mark cannot leak into a different group.

**Layer 5 – decide which groups are in danger.** A stone with no "safe" mark within three steps along its group is treated as in atari. Another flag means "2 liberties or fewer". The atari detection never misses a real atari. It can wrongly flag a very long group whose other liberties are further than three steps away.

**Layer 6 – score features for each point.**
- **Group status:** playing next to an opponent group in atari gives +12 (capture). Playing next to an opponent group with 2 liberties gives +3 (atari). Saving your own group from atari gives +8, and adding a liberty to a 2-liberty group gives +2.
- **Self-atari:** a move that leaves its stone with one liberty costs −15, or −30 when the point has no empty neighbour. This penalty is cancelled when the move captures.
- **Own eyes:** filling one costs −30, which is below passing.
- **Cut and connect:** on each 2×2 square, it adds +4 for connecting two diagonal stones that the opponent threatens to cut, and for cutting the opponent's diagonal pair the same way. This covers 52 of the 64 squares (the four corner squares and the edge squares next to them are left out).
- **Influence:** a smoothed count of own minus opponent stones nearby. Points already firmly yours are penalised, as are points deep in the opponent's area, while open and contested points score higher.
- **Position and shape:** a fixed preference by line (first and second lines cost more early in the game), a small bonus for the centre point (4,4), and bonuses for diagonal and one-space shape and for contact with opponent stones.
- **Pass:** its logit starts very low and rises with the number of stones on the board. The network passes only when every remaining move is clearly useless or harmful.

## Why I expect it to win

The opponents play with no search and sample their moves, so they regularly leave groups in atari or play self-atari. This network almost always captures those groups, including large ones, and it rarely makes the same mistakes itself. That is why its wins are mostly large: about 52 opponent stones captured per game against "strong", against about 10 of its own lost. The evaluation opponents beat a random player only 79–98% of the time, so most are weaker than "strong" and should be beaten even more often.

## Remaining weaknesses

- **Close games:** games decided by a few points are roughly 50/50, because its whole-board strategy is crude.
- **Long groups:** a very long group whose liberties are far away can be mistaken for one in atari. The network then plays a "capture" that is actually self-atari.
- **Repetition losses:** a few games per hundred are lost to the rule that repeating an earlier position loses. This usually follows one of those false captures, when the network recaptures in a way that repeats the board.
- **Score trend:** results against "strong" drifted from 76% (passes 4–5) to 69–73% in later versions, and I could not pin down which change caused it. The differences are within noise. I kept the final version because it alone was tested in exactly this form, and it is better at capturing large groups.