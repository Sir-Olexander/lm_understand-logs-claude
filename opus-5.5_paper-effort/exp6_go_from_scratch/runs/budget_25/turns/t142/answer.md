The final network is saved. On the last checks it won about 72% of games against the "strong" practice opponent (138 of 192), 100% against "weak" and 100% against random.

**Results**

| Practice opponent | Games | Win rate |
|---|---|---|
| strong | 64 + 128 (two separate passes) | 66% and 75%, about 72% combined |
| weak | 32 | 100% (earlier versions 94–100%) |
| random | 32 | 100% |

The hidden opponents beat random 79–98% of the time, so I'd expect the trained-opponent average to be at least the ~72% seen against "strong"; it should be higher if most of them are closer to "weak", but that is an extrapolation, not something I measured. The final version is better as White (75–80% in the earlier pass) than as Black (64–72%).

Many tested variants (77%, 75%, 73%, 70%, 68%) sit within about one standard error (±4–5% per 96–128 games) of each other. The final choice is one of the best-measured versions rather than a clearly superior one.

**How the network works**
- **Board copy:** each layer carries one unit per point that encodes empty, own stone or opponent stone. It tells each layer which stones belong to which side.
- **Liberty tracking:** liberties are marked with colours and spread one step per layer through connected stones, so a group knows which liberty colours it can reach.
  - My groups use 3 colours, tracked 4 steps. This tells a group in atari (1 liberty) apart from groups with 2 and with 3 or more.
  - Opponent groups use 2 colours (point parity), tracked 5 steps. A small extra check catches a common edge shape where both liberties have the same colour.
- **Move scores:**
  - **Capture:** +20 per adjacent opponent stone whose group is in atari.
  - **Escape:** +16 per adjacent own stone in atari.
  - **Self-atari:** a penalty when the move would leave the merged group with one liberty. It is switched off when the move captures.
  - **Eye filling:** −40.
  - **Position:** a centre-first preference by line, a bonus for points where both sides have nearby stones, and a small bonus for each empty neighbour.
  - **Pass:** starts at −12 and rises slightly as the board fills, up to a cap. That way we pass only when the remaining moves are bad, and still capture when far ahead.
- **Check:** I confirmed with three positions passes that every key unit matched what the Go rules say it should be.

**Why I expect it to win:** the opponents sample their moves, so they often leave groups in atari or play into self-atari. This network punishes that almost every time; about half the wins against "strong" ended with the opponent having nothing left on the board. It also rarely loses its own groups, never fills its own eyes, and plays its top positional move most of the time.

**Remaining weaknesses**
- **Big groups misread:** an opponent group too large for the 5-step check can look capturable when it isn't. This sometimes leads to playing my own stones into atari.
- **Repetition losses:** the network can't see past positions, so it sometimes loses by repeating a position. This most often follows sacrificing two stones and recapturing one, and cost about 3–6% of games even when ahead. A targeted fix cut these losses in one pass of 128 games but not in the next 64. Its combined result was lower (about 70%), so I left it out.
- **Territory:** when the opponent avoids big tactical mistakes, it usually wins the territory battle by a few points, especially when I'm Black.