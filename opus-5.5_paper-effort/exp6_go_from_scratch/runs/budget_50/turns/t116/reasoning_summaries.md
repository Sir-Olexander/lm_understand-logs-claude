I'm noting that the four repetitions all happened while I was ahead, so I lost four games, about 3%. The isolated single-capture pattern weakened with the lower scap3 value, but since the alternatives are weak late in the game, it still gets chosen almost every time. I'm considering a targeted fix for this pattern, which usually shows up in the late game.

I'm weighing a targeted penalty for isolated single-stone captures in the late game, since those usually sit inside the opponent's territory and get recaptured. Raising the weight on the M unit would catch them but also punish ordinary two-liberty moves too hard, so the adjustment has to be narrower.

I'm checking whether scaling the capture term down for the single-stone case would avoid penalizing it, and it looks like with the adjusted weight the isolated capture scores zero, so it wouldn't be penalized at all. The ko capture would get hit, which may be fine, while captures at eyes stay unpenalized. I still have 23 spare units in L6 to work with, so the fix is cheap enough to try.

I'm weighing whether a dedicated unit for this case fits, but L6 only has 23 spare units against the 81 it would need. Instead I'm looking at whether the existing 81-unit Terr or Eye could merge, since both penalize own-territory-like points.

I'm now testing a combined unit that could cover both the eye case and the territory case, using a stronger adjacency weight so that an interior eye gets penalized while a point with only two own neighbours does not.

I'm now considering whether to drop the repetition fix into the pass logit instead, since these cycles tend to show up late in dead areas.

I'm weighing a pass-logit adjustment instead: when far ahead, a higher pass weight would make passing competitive with the repeating capture move. The risk is that passing gets chosen too often when the estimate is overly optimistic in the middle game, so the early-game phase term has to keep it suppressed.

I'm considering a nonlinear pass adjustment, adding a single cheap unit that only boosts the pass logit when the estimated lead is above 15, so passing becomes competitive only when far ahead.

I'm weighing whether a pass-logit bump only when the estimated lead exceeds 15 would be cheap, but passing while the opponent keeps invading my territory is a real risk. Responses to threats would still outweigh the pass, while ordinary contact replies would not. The repetition cost is only about 3%, so this may not be worth the danger. I'll check the weak opponent and random baselines under the current config first, to confirm there's no regression before changing anything else.

I'm wondering whether the opening is the bigger lever, since black at about 45% versus white at about 65% suggests komi 7.5 is hard to overcome on 9x9. I'll check what the RL opponents play as white against our black, starting with how our first moves look under the early prior, where tengen should be the most likely choice.

