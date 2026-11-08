I'm looking back at game 11, which we lost by repetition even though we were ahead by 40.5 points. The end involved a ko-like cycle on the left side, where we kept playing inside their area and they kept capturing, until a move repeated an earlier position under the 8-board history check. I'm wondering how to avoid getting into that loop, and one idea is to stop playing single stones there.

I'm weighing a pass rule: when the stone difference is large and positive, boost the pass logit so we stop making invasive moves, though tactical moves should still outrank it. I'm noting that stone count doesn't reflect territory, so the threshold would need checking against the data.

I'm working out a way to make the pass signal depend on the stone difference: when we're far ahead, a global value computed from stone counts would boost the pass logit. I'm checking whether that value can be carried through the layers, since the last layer has no spare capacity and would need a point or two removed from its subset.

I'm tallying how often the repetitions happened across the games: nine ended that way, and in eight of them our move closed the cycle. I'm suspecting we keep recapturing greedily where the RL bot holds back, and I'm listing the longer cycle types, like sending-two-returning-one, triple ko, and eternal life, to see which ones we need to avoid.

I'm tracing game 3 next: after the opponent captures at 26, we replay 17 and then 8, and that recreates an earlier board. It looks like a throw-in, capture, and retake cycle that repeats whenever the opponent passes between moves. The fix seems to be avoiding self-atari moves inside their area.

I'm looking at the corner move at (0,8): it has only one empty neighbor, so the self-atari penalty should apply, which makes me wonder why we still played there. It could have been a legitimate capture of the stone at 26, but then the follow-up at 8 is unclear. I'm starting to think the cleaner fix is to pass when the opponent has passed and we have nothing useful, since our pass logit is strongly negative and the positional value of moves inside their territory is close to zero.

I'm weighing whether the pass logit should rise as stones accumulate late in the game, since passing would be fine once only neutral dame or dead invasions remain. The worry is passing while real moves are still left, though if the opponent passes too, the game ends and those remaining moves are worth close to nothing. I also notice the positional term gives a bonus for each adjacent opponent stone, which seems to reward invading deep inside their territory and could be what keeps feeding the cycles.

I'm now thinking about whether a position inside the opponent's territory can be recognized by a linear rule: a point whose neighbors are all opponent or empty, with no friendly stones close by, would be an invasion to discourage. The condition "no own stones within distance 2" isn't linear, so I'm considering penalizing nearby opponent stones while rewarding nearby own stones instead. I'm sketching a positional function built from two units per point at the last layer, after the eye feature is merged into the first.

I'm sketching a second unit per point that penalizes areas where opponent influence outweighs ours by more than a threshold, so the network avoids playing deep in their territory. Our own dominated areas would be wasteful too, but a purely linear penalty would also hit normal boundary moves. Writing the contested-area value as the minimum of the two influences can be done with a linear term plus one such unit.

I'm settling on a value term for each point: a bonus scaled by the smaller of the two influences, minus a penalty on their sum, plus the line value. That rewards contested boundary areas and gives nothing deep inside either side's territory. I'm still working out the kernel weights at distances 2 and 3.

I'm checking the sanity cases now: on an empty board the center point should win on line value alone, and after one stone each the contested points between them should score highest. I'm also wondering whether a small bonus for contact with an opponent stone should be kept, and I need to settle the value of the spacing parameter μ against crowded areas.

I'm settling on a value formula for each point: λ times the smaller of the two influences, minus μ times their sum, plus the line value. I'm leaning toward λ = 3 and μ = 0.5, and I'm checking how the boundary, open, and deep-territory points score under those weights.

I'm settling the endgame case too: dame between our and their stones sit on the boundary and score high, so they get filled, while points deep inside the opponent's territory come out negative, which should make them less attractive than passing.

I'm settling on a late-game pass logit that rises with the number of stones on the board, so that once the board fills, passing beats the invasion and own-territory moves that keep feeding repetition cycles. Boundary and tactical moves would still win over passing at that point.

I'm planning to add a pass-logit term that grows with the stone difference, and I need two spare units at layer 5, so I'm dropping two more points from its subset to make room for the global stone-count and lead features.

I'm now writing out the concrete version of the positional unit: the value of each point combines a term scaled by the smaller of the two influences with a penalty on their sum, and I'm working out the constants so the bound on the linear unit holds.

I'm working through the constants for the positional term: the bias C should be large enough to keep the first unit positive, and the output is P1 minus C minus λ times P2. I'm also noting that the global stone and difference units feed the pass logit, and I'll tighten the scale and retune the line values. Next I'll check the kernel and whether the min-of-influences term behaves well by running a quick test.

