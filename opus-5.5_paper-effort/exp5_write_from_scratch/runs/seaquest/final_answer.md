## Design

**Strategy.** The sub stays in columns 5–6. It only moves sideways to turn around, which shifts the front cell by one. It moves vertically toward the nearest rows that hold enemies or divers (the reward channel) and fires at enemies in its own row. It does not enter a row whose centre is occupied by danger. It dodges enemy bullets in its own row. It surfaces when oxygen is below 60 and it carries a diver. It never moves up into row 1 unless surfacing, because a sticky "up" there would surface with no diver and end the episode.

**Hidden layer 1 (32 units).** Every per-row unit is gated by the sub's cells in that row. When the sub is elsewhere the unit saturates at −1, so the second layer can read the value for the sub's row from a plain sum.
- **u0–6, V_r (rows 2–8):** signed vertical pull. It sums reward cells in the other rows (rows 2–8), weighted by closeness. Targets above count positive (up), targets below count 0.85× as much and negative.
- **u7–13, H_r (rows 2–8):** horizontal targeting, z = (reward cells left, cols 0–4) − (reward cells right, cols 7–9) + 0.3·F. F = +1 if the sub faces left; it is computed from the back cell, which is the only sub cell that is also in channel 4.
  - |z| ≈ 1.3 means an enemy is in front: fire.
  - z ≈ ±0.7 means an enemy is behind: turn.
  - |z| ≈ 0.3 means no target in the row.
- **u14–22, Bl_r (rows 0–8):** blocking. Danger in columns 3–8 of the row above gives +; danger in the row below gives −.
- **u23–29, Y_r (rows 2–8):** own-row enemy bullet nearby (danger cell that is not a reward cell, columns 2–9). These units trigger a dodge.
- **u30, Sf:** oxygen-gauge column 2 is off (oxygen below 60) and column 8 is on (at least one diver). This means "surface".
- **u31, R12:** the sub is in row 1 or 2.

**Hidden layer 2.** These units threshold the gated sums:
- vertical: weak up, weak down, strong down (the strong-down signal makes the sub dive from rows 0–1);
- horizontal: fire-left, fire-right, turn-left, turn-right;
- blocking: up blocked, down blocked;
- dodge: dodge from rows 2–7, dodge from row 8;
- surfacing: "row 1–2 and not surfacing" (blocks up), and a copy of the surfacing signal.

**Logit priorities:**

| Behaviour | Logit |
|---|---|
| Blocked or forbidden move | −45 |
| Surface, or dodge | ≈24–34 |
| Turn | 17 |
| Fire at an enemy in the row | 16 |
| Weak vertical move | 10 |
| Idle fire | 6 |

```edits
# ---------- Layer 1: V units (u0..u6 = rows 2..8), vertical attraction ----------
W1[120:180:6, 0] = 6
W1[180:240:6, 1] = 6
W1[240:300:6, 2] = 6
W1[300:360:6, 3] = 6
W1[360:420:6, 4] = 6
W1[420:480:6, 5] = 6
W1[480:540:6, 6] = 6
b1[0:7] = -12
# u0 (row 2)
W1[182:240:6, 0] = -0.51
W1[242:300:6, 0] = -0.38
W1[302:360:6, 0] = -0.30
W1[362:420:6, 0] = -0.24
W1[422:480:6, 0] = -0.19
W1[482:540:6, 0] = -0.15
# u1 (row 3)
W1[122:180:6, 1] = 0.6
W1[242:300:6, 1] = -0.51
W1[302:360:6, 1] = -0.38
W1[362:420:6, 1] = -0.30
W1[422:480:6, 1] = -0.24
W1[482:540:6, 1] = -0.19
# u2 (row 4)
W1[122:180:6, 2] = 0.45
W1[182:240:6, 2] = 0.6
W1[302:360:6, 2] = -0.51
W1[362:420:6, 2] = -0.38
W1[422:480:6, 2] = -0.30
W1[482:540:6, 2] = -0.24
# u3 (row 5)
W1[122:180:6, 3] = 0.35
W1[182:240:6, 3] = 0.45
W1[242:300:6, 3] = 0.6
W1[362:420:6, 3] = -0.51
W1[422:480:6, 3] = -0.38
W1[482:540:6, 3] = -0.30
# u4 (row 6)
W1[122:180:6, 4] = 0.28
W1[182:240:6, 4] = 0.35
W1[242:300:6, 4] = 0.45
W1[302:360:6, 4] = 0.6
W1[422:480:6, 4] = -0.51
W1[482:540:6, 4] = -0.38
# u5 (row 7)
W1[122:180:6, 5] = 0.22
W1[182:240:6, 5] = 0.28
W1[242:300:6, 5] = 0.35
W1[302:360:6, 5] = 0.45
W1[362:420:6, 5] = 0.6
W1[482:540:6, 5] = -0.51
# u6 (row 8)
W1[122:180:6, 6] = 0.18
W1[182:240:6, 6] = 0.22
W1[242:300:6, 6] = 0.28
W1[302:360:6, 6] = 0.35
W1[362:420:6, 6] = 0.45
W1[422:480:6, 6] = 0.6

# ---------- Layer 1: H units (u7..u13 = rows 2..8), aim/turn ----------
W1[120:180:6, 7] = 6
W1[144:163:6, 7] = [6.45, 6.15, 5.85, 5.55]
W1[148:167:6, 7] = [-0.9, -0.3, 0.3, 0.9]
W1[122:147:6, 7] = 1
W1[164:180:6, 7] = -1
W1[180:240:6, 8] = 6
W1[204:223:6, 8] = [6.45, 6.15, 5.85, 5.55]
W1[208:227:6, 8] = [-0.9, -0.3, 0.3, 0.9]
W1[182:207:6, 8] = 1
W1[224:240:6, 8] = -1
W1[240:300:6, 9] = 6
W1[264:283:6, 9] = [6.45, 6.15, 5.85, 5.55]
W1[268:287:6, 9] = [-0.9, -0.3, 0.3, 0.9]
W1[242:267:6, 9] = 1
W1[284:300:6, 9] = -1
W1[300:360:6, 10] = 6
W1[324:343:6, 10] = [6.45, 6.15, 5.85, 5.55]
W1[328:347:6, 10] = [-0.9, -0.3, 0.3, 0.9]
W1[302:327:6, 10] = 1
W1[344:360:6, 10] = -1
W1[360:420:6, 11] = 6
W1[384:403:6, 11] = [6.45, 6.15, 5.85, 5.55]
W1[388:407:6, 11] = [-0.9, -0.3, 0.3, 0.9]
W1[362:387:6, 11] = 1
W1[404:420:6, 11] = -1
W1[420:480:6, 12] = 6
W1[444:463:6, 12] = [6.45, 6.15, 5.85, 5.55]
W1[448:467:6, 12] = [-0.9, -0.3, 0.3, 0.9]
W1[422:447:6, 12] = 1
W1[464:480:6, 12] = -1
W1[480:540:6, 13] = 6
W1[504:523:6, 13] = [6.45, 6.15, 5.85, 5.55]
W1[508:527:6, 13] = [-0.9, -0.3, 0.3, 0.9]
W1[482:507:6, 13] = 1
W1[524:540:6, 13] = -1
b1[7:14] = -12

# ---------- Layer 1: Bl units (u14..u22 = rows 0..8), blocked above(+)/below(-) ----------
W1[0:60:6, 14] = 8
W1[79:110:6, 14] = [-2, -4, -4, -4, -4, -2]
W1[60:120:6, 15] = 8
W1[139:170:6, 15] = [-2, -4, -4, -4, -4, -2]
W1[120:180:6, 16] = 8
W1[79:110:6, 16] = [2, 4, 4, 4, 4, 2]
W1[199:230:6, 16] = [-2, -4, -4, -4, -4, -2]
W1[180:240:6, 17] = 8
W1[139:170:6, 17] = [2, 4, 4, 4, 4, 2]
W1[259:290:6, 17] = [-2, -4, -4, -4, -4, -2]
W1[240:300:6, 18] = 8
W1[199:230:6, 18] = [2, 4, 4, 4, 4, 2]
W1[319:350:6, 18] = [-2, -4, -4, -4, -4, -2]
W1[300:360:6, 19] = 8
W1[259:290:6, 19] = [2, 4, 4, 4, 4, 2]
W1[379:410:6, 19] = [-2, -4, -4, -4, -4, -2]
W1[360:420:6, 20] = 8
W1[319:350:6, 20] = [2, 4, 4, 4, 4, 2]
W1[439:470:6, 20] = [-2, -4, -4, -4, -4, -2]
W1[420:480:6, 21] = 8
W1[379:410:6, 21] = [2, 4, 4, 4, 4, 2]
W1[499:530:6, 21] = [-2, -4, -4, -4, -4, -2]
W1[480:540:6, 22] = 8
W1[439:470:6, 22] = [2, 4, 4, 4, 4, 2]
b1[14:23] = -16

# ---------- Layer 1: Y units (u23..u29 = rows 2..8), own-row enemy bullet ----------
W1[120:180:6, 23] = 10
W1[133:180:6, 23] = 4
W1[134:180:6, 23] = -4
W1[180:240:6, 24] = 10
W1[193:240:6, 24] = 4
W1[194:240:6, 24] = -4
W1[240:300:6, 25] = 10
W1[253:300:6, 25] = 4
W1[254:300:6, 25] = -4
W1[300:360:6, 26] = 10
W1[313:360:6, 26] = 4
W1[314:360:6, 26] = -4
W1[360:420:6, 27] = 10
W1[373:420:6, 27] = 4
W1[374:420:6, 27] = -4
W1[420:480:6, 28] = 10
W1[433:480:6, 28] = 4
W1[434:480:6, 28] = -4
W1[480:540:6, 29] = 10
W1[493:540:6, 29] = 4
W1[494:540:6, 29] = -4
b1[23:30] = -22

# ---------- Layer 1: surface-needed (u30), rows 1-2 indicator (u31) ----------
W1[593, 30] = 4
W1[557, 30] = -4
b1[30] = -2
W1[60:180:6, 31] = 3
b1[31] = -3

# ---------- Layer 2 ----------
W2[0:7, 0] = 60
b2[0] = 357
W2[0:7, 1] = -60
b2[1] = -363
W2[0:7, 2] = -60
b2[2] = -415.8
W2[7:14, 3] = 30
b2[3] = 157.5
W2[7:14, 4] = -30
b2[4] = -202.5
W2[7:14, 5] = 30
b2[5] = 166.5
W2[7:14, 6] = -30
b2[6] = -193.5
W2[14:23, 7] = 20
b2[7] = 150
W2[14:23, 8] = -20
b2[8] = -170
W2[23:29, 9] = 3
b2[9] = 15
W2[29, 10] = 5
W2[31, 11] = 3
W2[30, 11] = -3
b2[11] = -3
W2[30, 12] = 3

# ---------- Output logits: 0 noop, 1 left, 2 up, 3 right, 4 down, 5 fire ----------
W3[3, 5] = 5
W3[4, 5] = 5
b3[5] = 16
W3[5, 1] = 8.5
W3[3, 1] = -8.5
W3[6, 3] = 8.5
W3[4, 3] = -8.5
W3[0, 2] = 5
W3[12, 2] = 12
W3[10, 2] = 12
W3[9, 2] = 7
W3[11, 2] = -22.5
W3[7, 2] = -22.5
b3[2] = -9
W3[2, 4] = 10
W3[1, 4] = 5
W3[9, 4] = 12
W3[12, 4] = -17.5
W3[8, 4] = -22.5
b3[4] = -13
```