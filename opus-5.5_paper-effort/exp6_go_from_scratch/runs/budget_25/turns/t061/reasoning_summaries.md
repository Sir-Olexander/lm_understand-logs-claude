Now I'm tracing the back-and-forth in game 19 with the opponent invading near the bottom-left, where we capture and they keep re-invading in a repeating cycle.

I'm working through the repeated cycle in the bottom-left, where the opponent's throw-in at (8,2) and my recapture at (8,1) would recreate an earlier board position and lose the game for me. Each round of invasion and capture repeats until whoever completes the cycle loses, and since the opponent's history-aware play avoids that, I'm the one who falls into it. Without board history I can't tell when a position has recurred, so I'm considering whether a different capture choice inside that eye space could break the loop.

Now I'm weighing whether to accept the repetition losses, which come to about 6% of games and roughly 3% that I might recover by changing the capture logic. Since close losses are more numerous, I'm wondering whether endgame correctness might be the better place to focus my effort.

Now I'm reconsidering the pass logit, which is currently too low to stop the agent from playing dame points late in the game. Interior points with the center line bonus might land near zero, so I'm thinking about shifting the pass value to around -12 + 0.4.

Now I'm weighing whether to drop the capture-recapture fight altogether, since passing a little later might cost less than the repetition risk from the invasions. I'm checking how the pass logit scales late in the game and whether a small line-1 and line-2 bonus would keep dame points from losing to passing.

Now I'm settling the endgame fix: dame points should always beat passing, so I'm checking that contested points still come out ahead of a pass once the late-game line adjustment is applied.

Now I'm comparing the scores as white versus black, 0.667 against 0.458, which means I need about eight more points to catch up as black. I'm also considering whether to tune the positional parameters, possibly sharpening λ to 4, and checking the average top-1 move probability in pass 8 for non-tactical moves.

