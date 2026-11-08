# Experiment 1: which game (open answer)

Correct: **4 / 4**

| Network | True game | Mapped guess | Correct | Confidence | `Game:` line |
|---|---|---|---|---|---|
| 0bbcc3ea | freeway | freeway | yes | 80 | frogger, a road-crossing game (freeway-style). |
| a9f535cc | breakout | breakout | yes | 80 | breakout (a mini breakout / brick-breaker on the 8×8 field). |
| bdf99279 | pacman | pacman | yes | 72 | pac-man (a mini maze game where you eat pellets while avoiding ghosts). |
| dfc6cb65 | pong | pong | yes | 75 | pong, single-player against a goal line on the right. the agent is the left paddle and the ball bounces around the 8 × 8 field. |

| Network | input | cache write | cache read | output | wall s | tool-free |
|---|---|---|---|---|---|---|
| 0bbcc3ea | 2 | 79462 | 0 | 12549 | 123.7 | True |
| a9f535cc | 2 | 79679 | 0 | 16179 | 133.2 | True |
| bdf99279 | 2 | 79599 | 0 | 12996 | 115.5 | True |
| dfc6cb65 | 2 | 79471 | 0 | 12009 | 112.7 | True |

Guesses mapped to `other` or `None` need a manual look (no keyword matched).
