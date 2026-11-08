I'm trying to tell real effects from noise here: five similar configs land anywhere from 50% to 63%, so 128 games each isn't enough to separate small differences. Pooling everything since pass 21 gives a rough mean near 54%, and I'm now thinking about which changes could plausibly move the win rate a lot, starting with captures, since winning games rack up about 85 captured stones.

I'm checking whether attacking weak opponent groups more aggressively could raise the win rate, since the captured-stone counts suggest high-capture runs win more. Captures per game look like a less noisy signal, but a few huge captures could still skew them, so I'm weighing whether tuning without a clearer signal is worth the passes.

I'm noticing that about 75% of available captures actually get taken, which leaves a quarter unused. Dead opponent stones left inside my territory would still count against me at game end, so those captures matter even if they can wait, and I want to sort the missed ones by the size of the capturable string and whether a different capture was the better move.

