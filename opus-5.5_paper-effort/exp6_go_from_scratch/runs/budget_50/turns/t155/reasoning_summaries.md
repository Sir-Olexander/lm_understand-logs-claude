I found 56% at c_open=0.5 was worse, so the best range looks like 0.25 to 0.35. I'll settle on c_open=0.3, with nine passes and seven code runs left. Next I'll build it, save it, and check it against the weak, random, and strong mix, and I'm weighing whether to add one more variant with c_opp at 1.3.

I'll test c_opp=1.3 alongside c_open=0.3, costing one run and one pass. If that doesn't beat the baseline, I'll revert to c_opp=1.0 with c_open=0.3, then run validation passes against the mixed opponents.

