I'm tallying how often each pass wins big and how often it loses to repetition, and noting that the big wins cluster around the strong-capture passes. In close games I win roughly 40 to 50 percent of the time. Pass 9 breaks down into 52 blowouts, 30 close games with 12 wins, 5 repetition losses, and 9 big losses.

I'm working through the repetition losses now: if those five games had been wins, pass 9 would sit near 74%. Close-game win rates are lower for passes 6 and 9 than for passes 4 and 5, so I'm weighing two fixes, cutting repetition losses and strengthening close games. For repetition, the chain is false captures leading to a two-stone self-atari, a recapture, then ko, and I'm considering whether a density-based term could reduce self-atari bypasses without flagging normal moves in dense areas as false self-ataris.

I'm now weighing a density-based term for the self-atari check, but it runs into trouble: for points with two empty neighbors, even a modest density weight would push their self-atari score positive and flag ordinary moves as false self-ataris. So I'd have to gate it on the empty-neighbor count, which makes it nonlinear. I'm also considering a different angle, avoiding the recapture itself by spotting the board pattern where capturing a lone opposing stone sets up the ko.

I'm checking whether the recapture can be spotted from the board itself: a lone opposing stone in atari next to an empty point that is walled in by opposing stones, which was where our captured stone used to be. I'm testing that pattern against a few of the games and it seems to hold in the cases I've checked.

I'm checking whether a recapture can be recognized from the board alone: a single opposing stone whose capture point sits next to an empty point that is surrounded by opposing stones, the kind of spot a captured stone would leave behind. I'm testing whether that pattern would also catch ordinary captures that next to an opponent's eye, since that could block normal play.

I'm weighing a way to catch the recapture pattern: a point next to the capture spot that is walled in by opponent stones except for that one spot. I'd count those near-eye neighbors around each empty point. It's only a proxy, since I don't have the move history, but it would flag the cases where the recapture starts the cycle.

I'm weighing whether to drop the CC units, which were added in pass 8 without clear evidence they helped, to free up 81 units for an opponent near-eye feature that flags empty points surrounded by opponent stones.

I'm now weighing a penalty on the capture point itself: an opponent near-eye unit would push down moves adjacent to it, though that also catches normal moves like attacking the eye, so it's too broad a fix.

I'm now looking at a way to make the penalty specific to the capture itself: gate the single stone's capture bonus on whether an opponent near-eye sits two steps away, since a flat penalty on every move beside that eye would be far too broad.

I'm checking whether a false capture can be told apart at points with zero empty neighbors. In the three PSK examples, the move is adjacent to one of our own stones, which has two liberties, and the rest of its neighbors are opponent stones. So the check would need to look at how many of our stones sit next to the move, not only at the empty neighbor count.

I'm checking whether the self-atari term can be relaxed for captures: the formula gives zero for the game 5 move because the adjacent opponent stone adds a penalty of 2. I'm considering whether the kill penalty should depend on how many of our own stones the move connects to, since the real question is how many liberties the merged group keeps after the capture.

I'm settling on a compromise for the kill term: one point per adjacent predicted-atari stone instead of two. With a single adjacent stone that gives a self-atari score of 1, so the penalty comes to about -3 once the capture bonus is added, which could go either way depending on the positional term. Two adjacent atari stones still bring it to zero.

I'm weighing a tweak to the self-atari term: counting own neighbors with two or fewer liberties as a small bonus, so the game-5 pattern tips slightly negative instead of zero, and I'm checking whether true captures still come out ahead.

I'm checking whether the self-atari term can stay sensible when the move touches a weak own stone: if the capture also rescues that stone, the escape bonus keeps the score positive, and in the PSK case there's no escape bonus, so it stays slightly negative.

I'm trying a different weighting for the self-atari term, dropping the per-liberty cost to -2·e while still adding a point for each own neighbor with two or fewer liberties. The worry is that this could block ordinary extensions from two-liberty groups, since with two empty neighbors the score would land at 1 and flag a safe move as self-atari.

I'm reworking the self-atari term so the constant starts at 3 and the penalty is -2 per liberty, which keeps a lone one-liberty stone flagged. The catch is that two weak own neighbors at two liberties would push a legal extension to a positive score, so I'm checking whether that case actually matters.

I'm tuning the self-atari term again, and the scaled version with a per-unit weight around 10 looks too fiddly to set blind. I'm going to step back and make one final decision with what I have.

I'm settling on a final configuration: the pass-9 setup with tengen back at 0.4 and the mixed F3 seeds, to reduce false self-atari flags. If the score holds at about 74% or better in pass 10, I'll keep it; otherwise I'll revert to the pass-5 or pass-4 config, which scored 76%.

I'm comparing the pass-4 and pass-5 configs, which both scored 76%, against a new setup that can only be tested once. A single 96-game run can't separate differences of two or three percent, so I'm weighing which design is better justified: binary big-capture scoring, exact seeds to cut missed self-atari flags, and a positional term whose scaling caused overrides in pass 4.

I'm settling the final design: exact seeds for F, the 'mix' seed option for F3, binary big-capture scoring, and the self-atari kill at 2. I'll use pass 10 mainly to check that nothing has broken.

I'm treating the CC units as the thing to give up, since their benefit was never shown. Freeing those 104 units for a per-point recapture feature would be a cheap way to target the repetition losses, but it trades away a term I can't justify keeping either way.

I'm considering whether the repetition losses could be reduced by penalizing the single stone that captured, since it has one liberty and sits beside an opponent eye. I'm sketching a unit that scores how closely the stone's surroundings match the recapture pattern, though it's hard to tell whether a few local counts would capture it.

I'm checking whether the straight-line pattern holds across all three recapture cases, with the stone sitting two points from the empty point at the end of the line, and wondering if that shape is just a coincidence of how those pairs formed.

I'm weighing whether to mitigate the repetition losses through the structure around the stone that made the capture. A per-stone unit tracking the straight-line recapture pattern looks too vague to build reliably with the features I have. I'm turning instead to the false-capture side, where the group is predicted to be in atari but may actually have more than one liberty.

I'm reconsidering whether the kill term could tell a true capture from a false one, since a real capture should free up liberties for our stone and a false one wouldn't. The catch is that a distant seed beyond reach 3 looks the same as no information at all, and I don't have spare budget in the feature set to track that. I'm going to set this idea aside and stick with the current design.

I'm dropping the opponent-side F3 layer and using the freed capacity for hash-based flags instead, which cut false-atari counts a bit. Wider reach would need more depth, so I'm not pursuing that, and I'm setting the false-atari problem aside for now. Next I'm settling on the F3 'mix' rule for the seeds.

I'm checking whether one linear threshold can cover both seed conditions: the exact rule needs 2e + Σ ≥ 5 and the old rule needs e + Σ ≥ 4. Taking the lower of the two thresholds at each value of e gives a combined cutoff of Σ ≥ 4, 3, 1 for e = 0, 1, 2, so I'm working out whether that union is clean enough to implement.

I'm checking whether a single linear threshold can capture both seed conditions, and with a weight of 1.5 on the empty-point count it works: the rule 1.5e + Σ ≥ 4 matches all the cutoffs I need for each value of e.

I'm checking that the combined rule matches the union of the two seed conditions, and it does at the boundary cases. Scaling by two gives integer weights, 3H + 2ΣH - 7, and the smallest positive values come out to 1 or 2.

I'm settling the rest of the weight scheme: the self-atari term stays at 15 for a single liberty, and tengen goes to 0.4. I'm also wondering whether the close losses come from passing too early, so I'll check a few of the near-miss games in pass 9 to see if dame were still left on the board.

