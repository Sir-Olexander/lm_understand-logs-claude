# Hand-built policy for MinAtar Asterix: hold column 5 and dodge vertically

## Strategy

Every enemy and every treasure crosses the whole board horizontally. So each one eventually passes column 5, which is where the player starts. The policy never moves left or right. It stays in column 5 and only chooses among stay, up and down, which turns the game into a 1-D problem over rows 1–8. Left, right and fire get a logit of −30.

Each row holds at most one entity, so a row's state can be read linearly from its cells.

- **Threat in row r** (staying in or entering (r,5) can kill this step): E(r,4) + E(r,5) + E(r,6) − Tr(r,5) equals 1 exactly in the dangerous case.
  - An enemy next to column 5 but moving away has its trail at column 5, so it cancels to 0.
  - An enemy at column 4 or 6 moving towards column 5 has its trail on the far side, so it stays at 1.
  - An enemy at column 5 always counts.
  - Checking only this is exactly safe for one step, at any enemy speed.
- **Soon threat in row r** (an enemy two cells away and approaching): E(r,3) + E(r,7) − Tr(r,4) − Tr(r,6).
- **Gold approaching in row r**: gold weights (c+1 on the left half, 10−c on the right half, 5 at column 5) minus trail weights (≈ position) give 1–2 when the gold still has to reach or is at column 5, and ≤ 0 otherwise.

## Hidden layer 1 (32 tanh units, gain 20, all saturated to ±1)

| Units | Meaning |
|---|---|
| 0–7 | Player is in row y (at column 5) |
| 8–15 | Threat in row r |
| 16–23 | Soon threat in row r |
| 24–31 | Approaching gold in row r |

## Hidden layer 2

**Units 0–23: action-safety units, one per (player row y, action).** The actions are stay, up and down; the target row is clamped to rows 1–8.
- Each unit is gated by the player-row unit. When the player is not in row y it saturates at −1, so it adds the same constant to every action.
- When active, it outputs a graded level for the target row:
  - threat → ≈ +1, which costs −40 logits;
  - soon threat → −7;
  - gold → +24 for stay, +8 for a move;
  - neutral → 0.
- Move units also look one row further in the same direction, at a penalty of −20 for a threat and −10 for a soon threat. This is because 10% of the time the next action repeats the previous one (sticky actions).

**Unit 24, "gold somewhere above".** It sums the per-row gold flags with exponential weights 2^(8−r) and compares the sum with a threshold set by the player's row. This exactly tests whether any approaching gold lies in a row above the player.
- **Unit 25, "gold somewhere below"**: the same construction, mirrored.

**Units 26–28, position preferences.**
- Row 1 adds a push down and row 8 adds a push up.
- A ±0.5 tie-break favours moves towards the centre.

## Logits

- Stay: +5 by default.
- Up: +15 when gold lies above. Down: +10 when gold lies below, so up wins ties.
- Danger penalties (−40) dominate everything else.

```edits
# ================= Layer 1 =================
b1[0:32] = -10
# units 0..7 : player in row y (col 5), y=1..8
W1[90, 0] = 20
W1[150, 1] = 20
W1[210, 2] = 20
W1[270, 3] = 20
W1[330, 4] = 20
W1[390, 5] = 20
W1[450, 6] = 20
W1[510, 7] = 20
# units 8..15 : immediate threat at (r,5): E4+E5+E6-Tr5
W1[85:98:6, 8] = [20, 20, 20]
W1[94, 8] = -20
W1[145:158:6, 9] = [20, 20, 20]
W1[154, 9] = -20
W1[205:218:6, 10] = [20, 20, 20]
W1[214, 10] = -20
W1[265:278:6, 11] = [20, 20, 20]
W1[274, 11] = -20
W1[325:338:6, 12] = [20, 20, 20]
W1[334, 12] = -20
W1[385:398:6, 13] = [20, 20, 20]
W1[394, 13] = -20
W1[445:458:6, 14] = [20, 20, 20]
W1[454, 14] = -20
W1[505:518:6, 15] = [20, 20, 20]
W1[514, 15] = -20
# units 16..23 : soon threat: E3+E7-Tr4-Tr6
W1[79:104:24, 16] = [20, 20]
W1[88:101:12, 16] = [-20, -20]
W1[139:164:24, 17] = [20, 20]
W1[148:161:12, 17] = [-20, -20]
W1[199:224:24, 18] = [20, 20]
W1[208:221:12, 18] = [-20, -20]
W1[259:284:24, 19] = [20, 20]
W1[268:281:12, 19] = [-20, -20]
W1[319:344:24, 20] = [20, 20]
W1[328:341:12, 20] = [-20, -20]
W1[379:404:24, 21] = [20, 20]
W1[388:401:12, 21] = [-20, -20]
W1[439:464:24, 22] = [20, 20]
W1[448:461:12, 22] = [-20, -20]
W1[499:524:24, 23] = [20, 20]
W1[508:521:12, 23] = [-20, -20]
# units 24..31 : gold approaching column 5 in row r
W1[62:120:6, 24] = [20, 40, 60, 80, 100, 100, 80, 60, 40, 20]
W1[64:120:6, 24] = [0, -20, -40, -60, -80, -180, -60, -40, -20, 0]
W1[122:180:6, 25] = [20, 40, 60, 80, 100, 100, 80, 60, 40, 20]
W1[124:180:6, 25] = [0, -20, -40, -60, -80, -180, -60, -40, -20, 0]
W1[182:240:6, 26] = [20, 40, 60, 80, 100, 100, 80, 60, 40, 20]
W1[184:240:6, 26] = [0, -20, -40, -60, -80, -180, -60, -40, -20, 0]
W1[242:300:6, 27] = [20, 40, 60, 80, 100, 100, 80, 60, 40, 20]
W1[244:300:6, 27] = [0, -20, -40, -60, -80, -180, -60, -40, -20, 0]
W1[302:360:6, 28] = [20, 40, 60, 80, 100, 100, 80, 60, 40, 20]
W1[304:360:6, 28] = [0, -20, -40, -60, -80, -180, -60, -40, -20, 0]
W1[362:420:6, 29] = [20, 40, 60, 80, 100, 100, 80, 60, 40, 20]
W1[364:420:6, 29] = [0, -20, -40, -60, -80, -180, -60, -40, -20, 0]
W1[422:480:6, 30] = [20, 40, 60, 80, 100, 100, 80, 60, 40, 20]
W1[424:480:6, 30] = [0, -20, -40, -60, -80, -180, -60, -40, -20, 0]
W1[482:540:6, 31] = [20, 40, 60, 80, 100, 100, 80, 60, 40, 20]
W1[484:540:6, 31] = [0, -20, -40, -60, -80, -180, -60, -40, -20, 0]

# ================= Layer 2 =================
# units 0..7 : STAY safety/gold for player row y (target row y)
W2[0:32:8, 0] = [10, 2, 0.0884, -0.3466]
W2[1:32:8, 1] = [10, 2, 0.0884, -0.3466]
W2[2:32:8, 2] = [10, 2, 0.0884, -0.3466]
W2[3:32:8, 3] = [10, 2, 0.0884, -0.3466]
W2[4:32:8, 4] = [10, 2, 0.0884, -0.3466]
W2[5:32:8, 5] = [10, 2, 0.0884, -0.3466]
W2[6:32:8, 6] = [10, 2, 0.0884, -0.3466]
W2[7:32:8, 7] = [10, 2, 0.0884, -0.3466]
b2[0:8] = -8.2582
# units 8..15 : UP for player row y (target max(1,y-1), lookahead y-2)
W2[0, 8] = 10
W2[8:32:8, 8] = [2, 0.0884, -0.1014]
W2[1, 9] = 10
W2[8:32:8, 9] = [2, 0.0884, -0.1014]
W2[2, 10] = 10
W2[9:32:8, 10] = [2, 0.0884, -0.1014]
W2[8:24:8, 10] = [0.2747, 0.1277]
W2[3, 11] = 10
W2[10:32:8, 11] = [2, 0.0884, -0.1014]
W2[9:24:8, 11] = [0.2747, 0.1277]
W2[4, 12] = 10
W2[11:32:8, 12] = [2, 0.0884, -0.1014]
W2[10:24:8, 12] = [0.2747, 0.1277]
W2[5, 13] = 10
W2[12:32:8, 13] = [2, 0.0884, -0.1014]
W2[11:24:8, 13] = [0.2747, 0.1277]
W2[6, 14] = 10
W2[13:32:8, 14] = [2, 0.0884, -0.1014]
W2[12:24:8, 14] = [0.2747, 0.1277]
W2[7, 15] = 10
W2[14:32:8, 15] = [2, 0.0884, -0.1014]
W2[13:24:8, 15] = [0.2747, 0.1277]
b2[8:16] = [-8.013, -8.013, -7.6106, -7.6106, -7.6106, -7.6106, -7.6106, -7.6106]
# units 16..23 : DOWN for player row y (target min(8,y+1), lookahead y+2)
W2[0, 16] = 10
W2[9:32:8, 16] = [2, 0.0884, -0.1014]
W2[10:24:8, 16] = [0.2747, 0.1277]
W2[1, 17] = 10
W2[10:32:8, 17] = [2, 0.0884, -0.1014]
W2[11:24:8, 17] = [0.2747, 0.1277]
W2[2, 18] = 10
W2[11:32:8, 18] = [2, 0.0884, -0.1014]
W2[12:24:8, 18] = [0.2747, 0.1277]
W2[3, 19] = 10
W2[12:32:8, 19] = [2, 0.0884, -0.1014]
W2[13:24:8, 19] = [0.2747, 0.1277]
W2[4, 20] = 10
W2[13:32:8, 20] = [2, 0.0884, -0.1014]
W2[14:24:8, 20] = [0.2747, 0.1277]
W2[5, 21] = 10
W2[14:32:8, 21] = [2, 0.0884, -0.1014]
W2[15:24:8, 21] = [0.2747, 0.1277]
W2[6, 22] = 10
W2[15:32:8, 22] = [2, 0.0884, -0.1014]
W2[7, 23] = 10
W2[15:32:8, 23] = [2, 0.0884, -0.1014]
b2[16:24] = [-7.6106, -7.6106, -7.6106, -7.6106, -7.6106, -7.6106, -8.013, -8.013]
# unit 24 : approaching gold in some row above the player
W2[24:32, 24] = [384, 192, 96, 48, 24, 12, 6, 3]
W2[0:8, 24] = [-900, -382.5, -190.5, -94.5, -46.5, -22.5, -10.5, -4.5]
b2[24] = -886.5
# unit 25 : approaching gold in some row below the player
W2[24:32, 25] = [3, 6, 12, 24, 48, 96, 192, 384]
W2[0:8, 25] = [-4.5, -10.5, -22.5, -46.5, -94.5, -190.5, -382.5, -900]
b2[25] = -886.5
# unit 26 : player in row 1 ; unit 27 : player in row 8
W2[0, 26] = 3
W2[7, 27] = 3
# unit 28 : player in upper half (rows 1-4)
W2[0:4, 28] = [3, 3, 3, 3]
b2[28] = 9

# ================= Output =================
W3[0:8, 0] = [-40, -40, -40, -40, -40, -40, -40, -40]
W3[8:16, 2] = [-40, -40, -40, -40, -40, -40, -40, -40]
W3[16:24, 4] = [-40, -40, -40, -40, -40, -40, -40, -40]
W3[24, 2] = 7.5
W3[25, 4] = 5
W3[26, 4] = 3.5
W3[27, 2] = 3.5
W3[28, 2] = -0.5
W3[28, 4] = 0.5
b3[0:6] = [5, -30, 11, -30, 8.5, -30]
```