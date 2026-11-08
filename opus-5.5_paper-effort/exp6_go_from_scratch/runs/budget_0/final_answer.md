My network is saved: all 14 tensors are set. It has not been played or run, so the expectations below come only from the design.

**Design: a hand-wired Go rule engine**

Every hidden unit is an exact logic gate on 0/1 values (or on small whole-number counts). Each layer also passes the board forward (own stones and opponent stones), so later layers can keep reading it.

| Layer | What it computes for every point |
|---|---|
| L1 | Copies of the board; "stone with no empty neighbour" for each colour; the number of empty neighbours of each stone. |
| L2 | Seeds for "this group has at least 2 liberties" for both colours: a stone with 2 or more empty neighbours, or two touching same-colour stones that each have one. Touching stones cannot share a liberty, so this is exact. Also seeds for "at least 3 liberties", worked out by counting liberties around a stone and its neighbours. |
| L3–L5 | The "safe" flags spread through connected stones of the same colour, up to 3 steps. A stone left flagged *unsafe* has no safe seed nearby. This catches every group in atari (no false negatives) and is exact for groups of 1–2 stones. L5 also computes a SOFT flag: "my stone here would end up on one liberty". |
| L6 | The final features listed below. |

The L6 features are:
- **Base score:** a position prior (centre first, the first line avoided early), own liberties, contact with opponent stones, and diagonal/jump shape.
- **Capture:** +80 for each adjacent opponent stone flagged unsafe.
- **Attack:** +12 for each adjacent opponent stone not known to have 3 or more liberties.
- **Escape:** +100 for each adjacent own unsafe stone, but only when extending really adds liberties.
- **Self-atari:** HARD and SOFT penalties.
- **Never fill your own eye:** −300.
- **Don't fill your own territory:** −150 for each own neighbour, when no opponent stone is within distance 2.
- **Forced capture:** +250 when all neighbours are opponent stones; such a move is only legal if it captures.

Pass is fixed at −20. That is below every move without a penalty and above eye-fills, own-territory moves and self-ataris. So the network keeps playing (filling neutral points, invading, capturing) and passes only when nothing useful is left.

**Why it should win**

- **It scores well under the end-of-game rules.** Scoring counts every stone left on the board, with no removal of dead stones. The network never fills its own eyes and keeps its territory empty. It hunts every enemy stone inside its area until captured, and it invades only where its stone keeps 2 or more liberties.
- **Its tactics are near-deterministic.** Logit gaps of 10 or more mean it almost always takes the capture, escape or atari when one exists. The RL opponents sample at temperature 1 and only beat random play 79–98% of the time. Against them, punishing every stone left in atari and every self-inflicted eye-fill should decide most games.
- **Its own groups stay hard to kill.** It escapes atari, refuses moves that would leave its own stones on one liberty, and keeps its eye space intact.

**Known weaknesses**

- **Liberty counts are approximate for large groups.** Big or snake-shaped groups that reach more than 3 steps from the nearest safe stone are guessed, so the network may treat a 2-liberty group as being in atari.
- **No ladder reading.** It will keep extending a group that is being chased in a ladder.
- **It cannot see game history.** It cannot detect the long-cycle repetition rule (any repeat loses) and cannot tell whether the opponent just passed.