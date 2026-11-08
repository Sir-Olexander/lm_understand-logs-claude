## Final network: a hand-built Go evaluator

Every weight was written by code from Go rules and hand-chosen formulas. Nothing was trained or fitted. The network has no search and no history, so each move is a one-ply judgement of the current board.

**Measured results with the saved weights** (last four practice passes):

| Practice opponent | Wins | Score |
|---|---|---|
| "strong" (beats random 98%) | 229 / 384 | ≈ 60% |
| "weak" (beats random 95%) | 109 / 112 | ≈ 97% |
| random | 16 / 16 (32 / 32 in an earlier pass on the near-identical previous setting) | 100% |

The hidden opponents beat random 79–98% of the time. Most of them should therefore sit between "weak" and "strong", so I expect a high win rate on average. Against strength like "strong" it should be roughly 60%, about 51% as Black (komi) and 69% as White.

### What each layer computes

All values are exact 0/1 or small whole numbers. For every point I keep a separate copy of these features for my stones and the opponent's.

- **Layer 1:** where my stones and the opponent's stones are (kept and passed forward through layer 5), and how many empty neighbours each stone has. It also starts a rough score estimate: whose turn it is (for komi) and empty points touched only by one colour.
- **Layer 2:** a local "this string has at least 2 liberties" proof, using `2·libs(q) + Σ libs of same-colour neighbours ≥ 3`. This test is always correct. It adds a looser "at least 3 liberties" proof, plus a total stone count used as a game-phase signal.
- **Layers 3–5:** these proofs are spread along connected stones of the same colour, one step per layer. A stone with no proof nearby is treated as "in atari"; one with only the 2-liberty proof is treated as "2 liberties". So each string is classed as 1, 2 or 3+ liberties.
- **Layer 6:** one more spreading step for my own strings. It then forms the move features below, and the score estimate drives the pass logit.

### What decides the move logits

- **Capture** opponent stones in atari: +12 per adjacent stone. Captures are where most wins come from — about 85 stones captured per won game.
- **Escape** from atari: +12. **Strengthen** my own 2-liberty strings: +6. **Atari** opponent 2-liberty strings: +6.
- **Self-atari penalty:** I estimate the liberties my stone would have after the move. Moves leaving 1 liberty get about −25 or more; moves leaving 2 get a milder penalty. The estimate counts a touching 2-liberty string only once, so moves that touch the same weak string twice are still caught. Single-stone ko-style recaptures are discouraged by the same penalty.
- **Eyes and own territory:** filling my own single-point eye costs −12, and points deep in my own territory get a graded penalty.
- **Position:**
  - Bonuses for contact with opponent stones, for touching or diagonal to my own stones, and for open space nearby.
  - A penalty on the first and second lines that fades as the board fills.
- **Pass:** strongly discouraged early in the game. Later it depends on the estimated score: I pass when clearly ahead and keep playing when behind.

### Known weaknesses

- **Repetition losses:** about 3–4% of games are lost because the network recaptures in a cycle and repeats a whole-board position. It cannot see game history, and my attempts to block this pattern cost more captures than they saved.
- **Liberty mistakes on large groups:** spreading the liberty proofs reaches only about 4 steps along a string, so very large groups are sometimes misjudged.
- **Playing Black:** komi still makes Black the harder side.