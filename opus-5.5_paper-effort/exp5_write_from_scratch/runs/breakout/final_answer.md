## Design

**Strategy: lead-tracking.** Each step the paddle moves toward the column the ball will occupy next. The ball moves exactly one column per step, so a paddle that sits under the ball stays under it. It catches the ball with one cell of slack on each side, which absorbs a single sticky-action error.

**Main tracker (most positions).** The predicted next column is `2·ball − trail`. Let `sA = 2·ballcol − trail' − paddlecol`. The paddle moves right if sA ≥ 1, left if sA ≤ −1, and does nothing otherwise.
- `trail'` uses the trail's column, except column 0 counts as −1 and column 9 counts as 10.
- These edge values make the paddle wait at the wall while the ball is pinned there for a step, then reverse immediately.
- When the paddle is already at the wall, a sticky repeat of the old move does nothing, so it costs nothing there.

**Bounce correction (ball and trail both in row 8).** This happens only right after a paddle hit, and here the trail can show the wrong horizontal direction.
- If the paddle is under the ball, the ball was deflected the other way and its next column is the trail column.
- If the paddle is under the trail, the ball kept its direction.
- One linear rule covers both cases: `sB = 2·ball + trail − 3·paddle`, with the ball counting as 1 at column 0 and 17 at column 9 for the edge cases.

**Hidden units.**
- **Layer 1:**
  - u0 and u1 are the right and left threshold detectors for sA: tanh(5·(±sA − 0.5)).
  - u2 and u3 are the same detectors for sB. They are gated by +300 per ball or trail cell in row 8 and −602.5 bias, so they output +1 only in the bounce case.
  - u4 is the bounce flag C (ball and trail both in row 8).
- **Layer 2:**
  - Right2 = tanh(4·(u0 − u4 + 2·u2 + 1)).
  - Left2 = tanh(4·(u1 − u4 + 2·u3 + 1)).
  - These choose the bounce detector when C is on and the main detector otherwise.
- **Outputs:** the right and left logits are 15·Right2 and 15·Left2. All other logits are 0, so the agent does nothing when neither detector fires.

```edits
# ---- unit 0 (R_A) / unit 1 (L_A): ball channel 3, column c weight +-10c
W1[9:600:60, 0] = 10
W1[15:600:60, 0] = 20
W1[21:600:60, 0] = 30
W1[27:600:60, 0] = 40
W1[33:600:60, 0] = 50
W1[39:600:60, 0] = 60
W1[45:600:60, 0] = 70
W1[51:600:60, 0] = 80
W1[57:600:60, 0] = 90
W1[9:600:60, 1] = -10
W1[15:600:60, 1] = -20
W1[21:600:60, 1] = -30
W1[27:600:60, 1] = -40
W1[33:600:60, 1] = -50
W1[39:600:60, 1] = -60
W1[45:600:60, 1] = -70
W1[51:600:60, 1] = -80
W1[57:600:60, 1] = -90
# trail channel 4, column c: -5*TA[c], TA = [-1,1,2,3,4,5,6,7,8,10]
W1[4:600:60, 0] = 5
W1[10:600:60, 0] = -5
W1[16:600:60, 0] = -10
W1[22:600:60, 0] = -15
W1[28:600:60, 0] = -20
W1[34:600:60, 0] = -25
W1[40:600:60, 0] = -30
W1[46:600:60, 0] = -35
W1[52:600:60, 0] = -40
W1[58:600:60, 0] = -50
W1[4:600:60, 1] = -5
W1[10:600:60, 1] = 5
W1[16:600:60, 1] = 10
W1[22:600:60, 1] = 15
W1[28:600:60, 1] = 20
W1[34:600:60, 1] = 25
W1[40:600:60, 1] = 30
W1[46:600:60, 1] = 35
W1[52:600:60, 1] = 40
W1[58:600:60, 1] = 50
# paddle row 9 channel 0
W1[540:600:6, 0] = [0, -5, -10, -15, -20, -25, -30, -35, -40, -45]
W1[540:600:6, 1] = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
b1[0] = -2.5
b1[1] = -2.5
# ---- unit 2 (R_B) / unit 3 (L_B): bounce case, gated on ball+trail in row 8
W1[483:540:6, 2] = [305, 310, 320, 330, 340, 350, 360, 370, 380, 385]
W1[484:540:6, 2] = [300, 305, 310, 315, 320, 325, 330, 335, 340, 345]
W1[540:600:6, 2] = [0, -15, -30, -45, -60, -75, -90, -105, -120, -135]
b1[2] = -602.5
W1[483:540:6, 3] = [295, 290, 280, 270, 260, 250, 240, 230, 220, 215]
W1[484:540:6, 3] = [300, 295, 290, 285, 280, 275, 270, 265, 260, 255]
W1[540:600:6, 3] = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
b1[3] = -602.5
# ---- unit 4 (C flag): ball in row 8 AND trail in row 8
W1[483:540:6, 4] = 10
W1[484:540:6, 4] = 10
b1[4] = -15
# ---- layer 2: select bounce detectors when C, else main detectors
W2[0, 0] = 4
W2[4, 0] = -4
W2[2, 0] = 8
b2[0] = 4
W2[1, 1] = 4
W2[4, 1] = -4
W2[3, 1] = 8
b2[1] = 4
# ---- outputs: right (3) and left (1); no-op actions stay at 0
W3[0, 3] = 15
W3[1, 1] = 15
```