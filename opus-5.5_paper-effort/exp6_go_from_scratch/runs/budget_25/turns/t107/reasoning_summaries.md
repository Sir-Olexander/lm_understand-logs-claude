I'm working out that many close losses end with dame still on the board, about one time in two the opponent passed first and we passed after, so each unfilled dame point is worth a point to us under area scoring. With -0.5 and 2 dame left, filling them could flip the result, and I'm checking pass 16 of game 8 as an example.

I'm now considering whether to fill dame instead of passing when the opponent passes first, since the pass logit late in the game can reach about +1 while a dame move may be unsafe or safe depending on its neighbours. I'm thinking about lowering the pass logit so it is chosen only when nothing else is left, because playing a safe move inside our own territory costs nothing under area scoring.

I'm looking at whether passing is ever better than a safe move under area scoring, since playing a harmless move costs nothing. The main exceptions I see are repetition risks in long endgames and self-atari or eye-filling moves. I'm considering lowering the pass boost (kS and kD) so that passing stays well below the dame moves, which should be positive in most positions.

I'm weighing whether filling our own territory could harm a group's eye space, since a two-point eye alone isn't enough to live. Self-atari is already caught by the Uc penalty, and the risk from eye-filling looks small, though not zero.

I'm lowering the pass boost so passing only wins when nothing safer is left, with kS at 0.2 and kD at 0.3, capping the pass logit near -5. The game can still end at the 162-move limit, which is fine. Invasion moves inside the opponent's territory are harmless in area scoring, and self-ataris stay penalised by the Uc term.

I'm deciding between bundling the pass fix with v9, which measured best at 75%, or with v12, which had fewer repetitions but a lower win rate of 71%. Since the changes in v12 are hard to separate, I'm leaning toward building v13 on v9 with the pass fix.

