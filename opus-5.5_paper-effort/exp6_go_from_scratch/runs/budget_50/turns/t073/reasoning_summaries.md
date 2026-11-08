The win rate fell to 38%, below the 48-50% I was getting before, and I'm losing more stones (22 versus 17.6). Savable situations rose sharply, so I'm getting into atari more often, and self-atari is still at 255 incidents.

I'm weighing whether to restore the attack term and the mild two-liberty penalty, since dropping them may explain why savable situations doubled. Restoring both won't fit in the layer budgets as they stand: the attack term needs the opponent's atari flag to propagate into layer 5, which only has 26 spare slots. I'm considering merging the two existing penalty units to free room.

I'm working out a combined penalty unit that could replace the two existing ones. The constraints require the activation to be positive when there are zero liberties and when one liberty is left with no threat, and zero otherwise, so I'm testing a threshold form with weights α and β and checking the inequalities against sample values like α = 0.9, β = 0.3.

I'm checking the threshold inequalities for the merged penalty: a value of 1 at zero liberties, 0.7, 0.4 and 0.1 as the threshold term grows, and 0.1 at one liberty with no threat, with everything else zero. That leaves a single unit with weight 250 that could replace the two existing ones, but I still need to confirm there are no other cases that break it.

I'm checking whether one unit can cover both the zero-liberty and one-liberty penalties, which would free room in layer 6 for a mild two-liberty penalty.

I'm now testing a single threshold form for the mild two-liberty penalty, with α = 1, β = 0.5 and c = 2.5, checking that it's positive at the right liberty and threat counts and zero elsewhere. The weight would then need to be set so the penalty comes out at the desired size.

I'm checking whether dropping NO6 would free enough room to move Terr out of layer 5, so the attack term's opponent-atari flag could fit there instead. The concern is that NO6's false capture flags may be what drives the self-atari incidents.

I'm weighing whether restoring the attack term is worth it, since the opponent-atari flag would need room in layer 5. I'm also thinking about whether NO6's capture flags are driving the self-atari incidents, and whether dropping NS6 or NO6 would free space. Maybe the drop to 38% is partly noise, but losing the attack term may be the real cause.

I'm trying to free space in layer 5 for the attack term, and the obvious candidates are the Terr unit or merging the two unit types. I'm now checking whether the US and UO units really must stay separate, since they have disjoint supports.

I'm checking whether a single combined unit could carry both the own-side and opponent-side signals, since their supports are disjoint. The catch is that the own-stone and opponent-stone counts need to be kept apart in the attack and defense terms, and summing them would mix them up.

I'm considering computing the attack term per point at layer 5 instead, summing the opponent-side unit values over each point's neighbours, which would avoid the propagation budget problem.

Deciding between dropping Terr or UO from layer 5: I'm leaning toward moving Terr into layer 6 by removing NO6 and using NO5 for the capture bonus instead, since NS6 guards against eye-filling mistakes.

I'm laying out the layer budgets for a rebuilt version, with the attack term and the mild penalty restored. The plan keeps the escape check for eye-filling and drops the capture check that caused false flags, which frees room for the territory unit in the last layer. The attack and defense weights are going to be reset, and the penalty weights need recalibrating.

