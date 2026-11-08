I'm treating the remaining "dame" points as illegal or self-atari, so the empties touching both colors aren't worth playing and those losses are purely positional. I'm at about 75%, and positional play will decide the games that don't end in a wipeout, with 8 passes and 27 code runs left. I'm now looking at options to strengthen positional play within that budget, starting with a cheap color-dependent LV idea.

I'm weighing a few ways to improve positional play within the remaining budget, but noise of about ±4% makes tuning by experiment hard with only 8 passes. I'm leaning toward a targeted tactical fix instead, since the S2R1 repetition losses are a known and fairly contained problem.

I'm weighing a targeted opening idea: having Black open at tengen instead of a random central point, since many 9x9 engines favor that. The edge-territory concern is also on my mind, with the third line possibly deserving more weight mid-game.

I'm considering a first move at tengen for Black, since it's a more balanced opening than the random central points I'm currently using. It would be cheap to add as a logit bonus on the first move, though the layer that feeds it would need one unit freed up, likely by shrinking one subset. I'm also wondering how to handle White's reply, since the opponent will probably open at tengen too.

Next I'm weighing whether a tengen opening would help, though the effect is probably small. Since I'm winning mostly through wipeouts, I'm considering pushing harder for fights by raising the contact weight, but my win rate as Black is already lower and I need more of those games. Before tuning anything, I want to check the lost games from pass 17 and count how many stones I lost to self-ataris versus other causes.

I'm going to check which self-ataris remain under v12 and compare them against v13 in pass 17, since the V-veto result is too noisy to judge on win rate alone.

