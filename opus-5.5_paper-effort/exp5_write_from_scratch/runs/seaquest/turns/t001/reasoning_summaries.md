I'm sketching a hand-built policy network for MinAtar Seaquest, aiming for a 2-layer MLP with 32 hidden units per layer, set by hand rather than trained. I'm also weighing what score that might reach, since learned DQN-style agents land somewhere around 10 to 30 and a hand-made policy that survives and shoots could collect a few points.

I'm checking the surface and oxygen rules for the policy: staying at row 0 is safe, but dropping to row 1 starts the oxygen clock, and each oxygen cell covers 20 steps. I'm also noting the enemy and diver spawn and move timings, and that the player's shot cooldown works out to one shot every 5 steps.

I'm weighing a simple strategy where the sub stays at the surface, since row 0 is safe from enemies and oxygen, but that earns nothing. Staying low and shooting would score points, though oxygen would run out after about 200 steps without divers to refill it.

I'm trying to recall the published MinAtar results for DQN on Seaquest, and I'm not certain of the exact figure. My rough memory is that it sits in the single digits to low teens, and that later PPO-style agents go much higher, which gives me a sense of how a hand-built policy might compare.

I'm settling on a plan where the sub stays in one row and fires steadily at whatever comes into that row, since enemies only move horizontally and so the danger is mostly from the sides. Enemies entering from either direction in rows 1 to 8 would be hit by a stationary shooter, so that's the basic approach I'll aim for, aiming for a few points per episode.

I'm weighing whether a stationary shooter can survive, since enemies arriving from behind would be fatal. Dodging by moving vertically looks more promising, because enemies only move horizontally and their positions are predictable. The real challenge is that a 2-layer MLP has to represent positions relative to the player, which is hard to do with a fixed set of weights.

I'm weighing a fixed-column design, where the sub keeps the same horizontal position and only moves vertically, so the MLP only has to read danger by row rather than by relative position. I'm thinking about whether a strategy like that could work with 32 hidden units per layer.

I'm weighing a hands-on approach where the sub patrols a column, dodging enemies by moving up and down while firing sideways at whatever enters its row. The catch is oxygen: once submerged with no divers, the sub has about 200 steps before it runs out, and resurfacing only helps if it brings at least one diver along.

I'm weighing whether to just stay at the surface, since there's no penalty for dying, so a score earned before oxygen runs out counts. The real aim is to kill as many enemies as I can in the first 200 steps, and pick up divers to resurface if the chance comes up.

I'm checking whether the sub could sit at an edge column, but enemies can spawn right on top of it there, so an edge position isn't safe either.

I'm checking the spawn timing: a diver or enemy can appear on the sub's own cell right when it spawns, so an edge column doesn't guarantee safety either.

I'm working out how turning works: at a wall, pressing toward it changes facing without moving, but elsewhere a turn always shifts the sub's column. Settling on a dodge-first approach that keeps the sub in one column and fires whatever comes into its row.

I'm checking how the danger works in our row: if an enemy sub is heading toward us, I'd move up or down to a safe row and fire whenever something lines up in front. Enemy bullets move a cell every frame, much faster than the subs themselves, so they're the real threat to dodge.

I'm checking how enemies threaten the sub in its own row: only those coming toward it matter, since an enemy moving away leaves its bullets heading away too. The trail behind each enemy shows its direction, though it's also used for divers and the sub's back cell. I'm considering treating every enemy in the row as a threat to keep it simple, accepting some over-dodging, and firing forward at anything headed my way.

I'm working out how a bullet and an enemy in the same row would meet: if the bullet is directly behind the enemy's next cell, they collide either way, so the shot connects. Enemy subs approaching in our row are the real danger, since they fire bullets every 11 frames that move toward us at one cell per frame, and our own shots can't block those.

I'm working out how an enemy sub's bullet would reach us: it spawns one cell ahead of the sub, then moves a cell per frame, so the bullet and our own shot pass each other rather than cancelling out.

I'm checking how an enemy bullet in our row would reach us: at one cell per frame, it's a threat that leaves only a step to react once it's adjacent. If it's coming straight at us, the only safe move is to leave the row, and a 10% chance that a sticky action repeats the previous move complicates a plan that depends on reacting one step at a time.

I'm weighing a fixed-column strategy where the sub stays put and only moves vertically to dodge, so it would face one direction and fire only at what enters that side. Keeping the column constant seems to simplify what the MLP has to represent, though it leaves the sub exposed to threats from the other side.

I'm weighing a sub at column 8 facing left, since an enemy spawning at column 9 would take about six frames to reach it, giving time to react.

I'm working out the timing of the enemy's movement: after spawning, it waits five steps before its first move, then steps once every six frames. That gives me roughly five steps to react to something entering from the side. I'm also noting that an enemy sub fires about ten frames after spawning, so its first bullet comes soon after the sub itself starts moving.

I'm checking how an enemy sub at column 0 moving right would threaten us at column 8 facing left, since its shot timer starts at 10 and its bullets would travel toward us one cell per frame.

I'm checking how the shot timing lines up with the enemy sub's first bullet: if I fire as soon as the sub appears, my shot should reach it around the time it first moves, before it has a chance to shoot. The concern is that repeated firing gives a shot only every five steps, so I need to think about whether a constant fire pattern would still leave me covered.

I'm settling on a fixed column so the policy stays simple, sitting near column 8 and firing at whatever approaches from the left while dodging vertically when threats close in from the right. I'm also noting that divers sliding through that column in their row could be picked up by moving onto them.

I'm now looking at whether a diver in column 8 could be targeted by moving vertically toward its row, which would mean comparing diver rows against my own row with a linear unit that tracks which divers sit above or below me.

I'm weighing whether the sub should hunt: moving vertically toward the row of the nearest enemy in front of it while dodging anything coming from behind.

I'm checking the edge case of sitting at column 9 facing left: the back cell would land at column 10, which is off the grid and could raise an index error, so that placement might crash the environment. That makes column 9 risky to rely on, and I'm weighing whether the sub should stay at column 8 instead.

I'm confirming that the walls protect against an index error: at column 9 the sub always faces right, and at column 0 it always faces left, so the back cell never leaves the grid. Those edge positions would still be useless for shooting, since the sub's gun points at the wall, which leaves me settling on column 8 facing left.

I'm weighing whether a fixed column is workable for the network: the sub would move right to column 9 and then back left once to reach column 8 facing left, or go left to column 0 and back right to column 1 facing right. The open question is whether the MLP can learn "move toward the target column and orientation" from the state alone.

I'm settling on column 8 facing left as the spot to hold, so I'd need to reach it from the surface by moving right to the wall and then back left once. The open question is how to tell from the observation whether the sub is already at column 8 facing left, since facing direction is what decides which way it shoots.

I'm now checking the row-0 logic: if the back cell sits at column 9 in the trail channel, the sub is at column 8 facing left, so it should dive. Otherwise it moves right, except when it's already at column 9 facing right, where it has to turn back left.

I'm settling the submerged plan: holding column 8 and moving only up or down, with left and right strongly suppressed so sticky or sampled actions can't drift the sub off its column.

I'm reconsidering whether column 8 is the right spot: enemy subs spawning at column 9 and moving left would reach column 8 after about five steps, so I'd only have a one-cell buffer against threats from behind.

I'm checking the threat from an enemy sub spawning at column 9 and moving left: it reaches column 8 within about five steps, so I'd need to dodge vertically before then. Since it's not drawn at column 9 until it appears there, I'd be reading that danger from the spawn position itself.

I'm checking the threat from an enemy sub at column 9 moving right: it exits after about six frames and its bullets head away from me, so it's harmless. Anything at column 9 in my row could still be moving left, so I'll treat it as a threat and dodge. From the left, an approaching enemy is something I can shoot, but an enemy sub already in the field could fire into my row before I get a shot off.

I'm checking which threats matter in my column: enemies and bullets coming from the right at column 9 or from the left, as well as anything drifting into the row I'm about to move to. Divers that cross column 8 are worth grabbing, since they stay there for about six frames as they move.

I'm settling the surfacing rule: when oxygen runs low and I'm carrying at least one diver, I'll head up to row 0, where oxygen refills and the difficulty ramps up a bit, then drop back down to column 8 facing left.

I'm working out how to read the oxygen and diver gauges from the bottom row: the oxygen bar fills columns 0 up to a count set by oxygen level, and the diver bar fills from column 9 leftward, so with one diver it's just column 8.

I'm checking the oxygen gauge against the diver gauge on the bottom row, since both draw into the same channel. Oxygen above 180 already lights column 8, and a full tank of 200 lights column 9 as well, so the two readings can overlap and hide a diver.

I'm working out how to tell from the bottom gauge row whether I have a diver while oxygen is low: below about 60 the oxygen bar only reaches columns 0 to 2, so a lit column 8 would have to come from the diver bar.

I'm working out whether the oxygen bar and the diver bar can be told apart on the bottom row: the oxygen fill covers the leftmost columns, so below about 80 oxygen a lit column 8 can only come from a diver.

I'm checking whether a low-oxygen threshold can be read from the gauge without the diver bar hiding it: column 2 can only be lit by oxygen below 60, so an unlit column 2 is a clean signal to surface.

I'm settling on oxygen below 60 as the surfacing trigger, since column 2 can't be lit by the diver bar, so it reads cleanly as low oxygen.

I'm working out the surfacing trigger: if I hit row 1 with no divers, going up would end the episode, so I'd dodge downward instead.

I'm settling on the surfacing trigger as low oxygen with at least one diver, and I'm weighing whether that threshold should be stricter or looser.

I'm working through how the 32-unit hidden layer would encode diver versus enemy cells, using channel 2 minus channel 1 as a rough diver signal, and I'm checking how the player's row can be represented relative to that.

I'm working out how the policy should choose between up, stay, and down by comparing safety and value across neighboring rows, and I'm considering whether a 2-layer MLP can express that with per-row features feeding into a player-row-conditioned combination.

I'm now weighing a per-row design for the 32-unit hidden layer: each unit would fire only when I'm in a given row and danger is present there, which lets the network tell whether moving up or down is blocked without needing a separate unit for every neighbor combination.

I'm thinking about giving each row its own unit that switches on only when the player sits in that row, using a large weight on the player-row feature so the unit gates on position. Inside that gate, a threshold over the neighboring rows' danger would indicate whether moving up or down is blocked. Summing these per-row units afterward would then let one hidden layer express the up/stay/down choice without a separate unit for every neighbor combination.

I'm weighing whether one gated unit per row can carry a thresholded feature of the neighboring rows' danger, which would give about four features per row within the 32 units. Since only the active row's unit contributes, summing them over rows could approximate a row-relative function without a separate unit for every combination. I'm also considering whether a second layer could recover more features from those.

I'm estimating the score a simple stay-and-shoot policy would get, and it comes out poor: enemies from the left in my row arrive only about 1 in 16 spawns, so in 200 steps I'd kill well under one. To do better, the policy has to move between rows to hunt enemies coming from the left.

I'm reasoning that a bullet keeps travelling along its row after I leave it, so I only need to fire once per target row. With a five-step cooldown, that caps me at about one kill per five steps, and I'm weighing how that compares with how fast enemies spawn.

I'm now looking at a hunting policy: move vertically toward the row of the nearest enemy on the left, since every enemy eventually crosses columns 0 to 7, and fire once I'm lined up with one.

I'm checking whether a policy that hunts enemies from the left, firing once per row and dodging threats, could reach around 10 kills in 200 steps. I'm also noting that enemy bullets already in flight still need the dodge logic, since firing at a sub in my row only stops it once the shot connects.

I'm checking how a bullet at column 7 or 8 interacts with my move into its row: if I step into a cell the bullet reaches on its next update, I'm dead. So I'd avoid entering a row with a rightward bullet close by, and I'm weighing whether to watch for bullets within two cells.

I'm noting that bullets leave no trail, so a danger cell alone doesn't reveal which way a bullet is heading. I'll treat any danger near column 8 in a row as blocking, and I'm weighing whether a fish at column 7 moving toward me in my row is a target to shoot rather than a threat to dodge.

I'm checking whether a shot fired from column 8 can hit a fish at column 7 in the same step, which would make firing a valid answer to an adjacent enemy. The catch is that I can't see the shot cooldown directly, so I'd have to infer it from the bullets already on screen.

I'm working out whether a friendly bullet's position tells me when my shot cooldown has expired: a shot fired from column 8 should be at column 3 once the five-step cooldown has run. So I'd treat the shot as ready when no bullet of mine sits in columns 4 to 7, though a bullet that has already hit something would vanish and leave that signal unclear.

I'm settling on column 8 as the firing spot, since a bullet from there reaches column 7 in the same step. Enemies appearing at column 9 and enemies drifting left in the row I'm entering are the threats to watch, so I'd treat any enemy at column 8 as deadly to step into.

I'm picturing the danger zone near column 8: cells at 9, 8 and 7 in a row, plus a bullet two cells out at column 6, all count as threats to that row. Enemies moving right from 7 or bullets at 6 could reach column 8 in a step or two, so entering that row would be risky.

I'm checking whether a fish at column 6 or a bullet there matters: enemies are fine to shoot, but a bullet at column 6 moving right reaches column 8 in two steps, so I need to tell bullets apart from fish, and I'm looking at the reward channel to do that.

I'm deciding whether an enemy sub at column 6 in my row needs to count as a threat, since a bullet it fires could reach column 7 and then hit me one step later. With sticky actions, a repeated move could also go wrong, so I'm weighing whether to watch column 6 as well.

I'm defining a row as "hot" whenever there's any danger in columns 6 to 9, and I'd dodge up or down out of a hot row. I'm also wondering whether firing at enemies on the left, once they're within columns 0 to 5, would still let me avoid getting caught.

I'm now going through which rows count as hot and which contain a target to hunt, using the reward channel to tell enemies from divers, since divers don't get shot.

I'm weighing how to pick a target row: danger in columns 0 to 5 probably means an enemy or an enemy's bullet, so I'd treat that row as worth hunting. There's still a problem, though. Once I fire at a target, its bullet needs several frames to reach it, so the target may still show up on the next steps and I'd end up firing again at something already being hit.

I'm now working out how to steer vertically: if there's a target in a row above, I'd move up, and if it's below, I'd move down. The trick is to only count a row as a target when it has danger in columns 0 to 5 and no friendly bullet already flying through it, so I don't keep firing at something already being hit.

I'm working out how to split the policy into separate per-row units: one that flags the current row as hot and needs a dodge, and others that compare target density above versus below the player's row to choose a vertical direction.

I'm trying to fold the player's row position into the per-row features, since the second layer can only see what the first layer outputs. Each row would need several flags (hot, blocked above or below, target direction, fire, diver attraction, oxygen), which is more than 32 units can hold, so I'm looking for a more compact encoding.

I'm thinking about gating the first layer on the player's row, so each unit computes a threshold over that row's neighborhood. With eight rows and four features per row, that fits in 32 units, and the second layer can sum across rows since only one gate is active at a time.

I'm now considering whether to split the first layer into row-pair units, where a unit fires when the player sits in one row and the row below it is hot, which would mark that downward move as blocked.

I'm thinking about whether to keep the player's column fixed at 8, since we never move horizontally underwater and sticky actions or random choices are the only things that could shift it. Using the front cell at column 8 as the gate for each row seems workable, so I'd only need to check whether a danger cell lies beside it.

I'm still sizing the per-row gates, so I'll check which decisions each row needs and whether they fit in 32 units. Each row needs to know if it's hot, whether moving up or down is safe, and which way the targets lie, and I'll see if that can be compressed.

I'm weighing a per-row gating scheme where each first-layer unit fires only when the player sits in one row, so the score for moving up versus down can be folded into a single unit per row. A separate unit would flag whether the current row is hot, and I'd combine them in the second layer to decide the vertical move.

I'm now setting the vertical logits as additive scores: moving up gets penalized by danger in the row above, moving down by danger in the row below, and both get a term that depends on whether my current row is hot. Firing is scored separately, with a penalty when the row is hot and a bonus when a target sits in it.

I'm turning to the per-row gated units: one per player row, each checking whether the row above is hot while the player sits in the current row. That's eight units for the upward check, and I'm noting that row 1 is special because its "up" is the surface, which is only safe when a diver is aboard.

I'm working out the surface rule for row 1: up is only safe when the oxygen gauge is low and a diver is aboard, which I'd capture as a threshold over the two gauge cells, then gate that on the player being in row 1.

I'm checking the first-row surface rule as a threshold: the surface counts as blocked unless the oxygen gauge is low and a diver is aboard. Working through the cases, that comes out right for the common ones, and the one ambiguous case, where the diver gauge is lit but oxygen is in the middle range, still needs a separate check.

I'm now checking the down-blocked units, one per row, which fire only when the player sits in row y and the row below is hot. That's eight more units, and the bottom row's down move is treated as blocked by default. Adding up the hot-row, up-blocked, and down-blocked units leaves only about eight units for targets, divers, oxygen, and firing, which is tight, so I'm wondering whether some of these checks can be merged.

I'm now looking at splitting the first layer into gated per-row units, and whether the second layer can recover the player's row from a set of "player row ≥ k" threshold units, since that would let it combine the row gate with the hot-row flags.

I'm weighing whether to encode the player's row as threshold units in layer one, so the second layer can gate the per-row hot and blocked checks on it. That would leave room for the target and diver preferences, though I'm not sure yet how to express the above-versus-below comparisons without spending too many units.

I'm weighing a layer-one unit per player row that adds up targets weighted by their distance from that row, so the sign of the sum says whether the targets lie above or below. The open question is whether gating each unit on the player's row can keep this within the 32-unit budget.

I'm working out a patrol strategy as a fallback to targeted hunting: sweep up and down through the rows continuously, and fire whenever the current row holds a target. Each row gets revisited every fourteen steps or so, and enemies only linger about 48 frames, so a sweep should still catch most of them.

I'm considering a sweep-based policy that doesn't need target-direction features, since patrolling every row would still catch enemies. The problem is that the MLP has no memory, so it can't tell which way it was moving. I'm wondering whether the oxygen gauge could serve as a clock, with its lit-cell count changing every 20 steps and its parity deciding whether to head up or down.

I'm checking whether the oxygen gauge's parity can act as a direction signal: an alternating sum over its lit cells would be linear and would flip with each 20-step change. The catch is that the diver bar overlaps the same columns once several divers are aboard, which could corrupt that sum.

I'm checking the oxygen gauge's parity over columns 0 through 6, since the diver bar only reaches columns 7 and 8 when there are one or two divers. When the oxygen level is at most 7 cells, that alternating sum should come out as 1 for an odd count and 0 for an even one. The open question is what happens once the oxygen count goes past 7.

I'm checking whether the oxygen parity signal stays clean when the count passes 7, and I'm noting that a patrol sweep would be simpler but walks into more danger than targeted hunting. Next I'm laying out a layer-one budget: per-row units for the player's row, for hot-row danger, and for the surface rule.

I'm considering merging the up-blocked and down-blocked units into one signed per-row unit that weighs safety and target pull together. The snag is that inactive rows saturate to −1 rather than 0, so I still need a way to make them drop out cleanly.

I'm now weighing how to encode the case where the player must leave a hot row and both the row above and below are blocked, since that's where a single per-row unit can't capture the answer on its own.

I'm settling the rule for the case where the player must leave a hot row and both neighbors are blocked, which is rare enough to leave to preference or firing. Otherwise, when the current row is clear and an enemy sits in it, I'd fire, and when it's clear without a target, I'd move toward the preferred direction if that row is free. I'm sketching the logits for up, down, fire, and noop with the blocking and must-leave terms as penalties.

I'm settling how the output layer gets its inputs: the up-blocked, down-blocked, and must-leave signals each get summed over the rows, and a layer-two unit can turn each sum into a clean plus-or-minus-one flag, which uses three units. That leaves the first layer with its 24 gated per-row units to work out.

I'm trying to fold the up-or-down choice into fewer units, since 24 gated per-row units won't fit alongside the target, fire, and row-0 logic. One option is to pool the "player in row y and row y±1 blocked" checks across rows, so a single unit can stand in for the whole set instead of one unit per row.

I'm moving the AND into layer two: layer one gets the eight hot-row flags and eight player-row indicators ungated, and then layer two can form the up-blocked, down-blocked, and must-leave conjunctions, which keeps the layer-one count at 16.

I'm trying to save units by folding the player-row indicators into the hot-row units, perhaps by gating them with a multi-level output, but I'm not sure that's workable yet.

I'm trying to pack the player-row indicator and the hot-row danger signal into a single layer-one unit per row, using a weighted sum that gives four distinct levels for the combinations of player row and danger. Layer two would then threshold those levels back apart.

I'm trying to get the same H_{y−1} hot-row signal to feed all three conjunctions at once, rather than building separate units for each. If one layer-1 unit could carry it, the layer-2 ANDs with the player-row indicators would stay simple, though I'm not yet sure a single tanh can keep the levels apart.

I'm weighing whether to compute the up, down, and must-leave conjunctions directly in layer one, which costs 24 units, or to stage them through layer two with 16 layer-one units, which would leave more layer-one room for the target and navigation logic.

I'm now thinking about how the target-attraction and fire-decision logic fits into the layer budget, since layer one is nearly full.

I'm weighing whether the target and preference logic can fit in the layer budget. Tracking each row's target and player position costs as much as the hazard units already do, so the network runs out of room. Trimming the rows the player can use would drop targets in the top and bottom rows, so that's not a clean option either. I'll look for features that can share units.

I'm trying to fold the row's target signal and its hot-danger signal into one layer-one unit per row, a signed value meant to read hot as −1, neutral as 0, and target as +1. A tanh over a weighted count of target cells minus a weighted count of hot cells could give that, with hot rows pushing the unit strongly negative.

I'm checking whether a tanh over the signed target-minus-hot count can encode both states in one layer-one unit, and then seeing whether the layer-two conjunction still separates the cases cleanly with the ±1 inputs.

I'm now weighing whether the player-row units can merge with the target units in layer one, which would let layer two combine them without running out of room. The layer-two budget is the real constraint: the up, down, fire, and target conjunctions already fill it, so I'm looking for a way to share one unit between the target and fire conditions.

I'm trying to fold the player-row and hot-row signals for each row into one layer-two unit, a value that reads +1 when the player is there and the row is clear, 0 when it's neutral, and −1 when it's dangerous. The catch is that for rows the player isn't in, the sum drops to about −1 regardless, so the unit only tells the cases apart within the active row.

I'm seeing that the signed sum over rows gives a single active-row value of −1, 0, or +1, which shifts the total into three distinct levels. That lets the fire logit scale with the target signal and the move logits scale against it, so hot rows push toward movement and target rows push toward firing.

I'm now seeing that a neighbor row's three-level state can be encoded the same way, so the up and down choices each get a unit whose sign tells me whether that adjacent row is hot, neutral, or a target. That also pulls the player toward target rows next door, which is useful. I'm noticing that the layer-two budget is nearly used up with 24 of these units, leaving about 8.

I'm checking the precision of the zero level in the signed hot-row unit, and whether a steep layer-one scale on the player-row indicator keeps its tanh output close enough to ±1 for the layer-two offsets to hold.

I'm working through how a signed unit per row can carry both the target and hot-danger states for the layer-two thresholds. The sums land near ±1 and 0 as intended, so I'm checking the scaling to keep those levels separated. I'm also considering whether a few broader layer-one units could pull the player toward farther targets without spending another layer-two unit per row.

I'm now looking at folding the diver pickup into the same per-row signal. A diver sitting in column 8 of a row, or just beside it, means moving into that row could collect it, unless the row is hot.

I'm now folding the diver signal into the same per-row unit, treating a diver in the player's path as a positive target that makes the player stay and wait for it to arrive. Diver presence is read as the reward channel minus the danger channel in each cell, so divers don't get confused with hazards.

I'm now folding the diver signal into the same per-row unit, so a diver in cols 7 to 9 adds positive weight while danger cells subtract, and I'm checking that the sum stays separate from bullet presence. I'm also weighing whether friendly bullets already in a target row should pull that row's targetness down, since the shot is already in flight and staying there wastes time.

I'm checking how a friendly bullet already in row r would affect that row's targetness, since subtracting it from the count would leave a single enemy row reading as neutral. I'm also noting that a second enemy would still keep the row as a target. Next I'm looking at whether enemies in columns 0 to 5 moving toward the sub should count as targets before they reach the danger zone.

I'm checking whether a bullet at column 5 moving right can be handled by the same target signal: it reaches column 6 in a step and turns the row hot, so I'd have only about one step to get out before it arrives. I'm timing the move so the sub clears the row before the bullet reaches column 8.

I'm making the bullet signal stronger in the outer columns: bullets get a negative weight and divers a positive one, so a bullet near the sub reads as danger while a diver reads as a pickup.

I'm weighing whether divers in columns 4 and 5 should count as targets, since a diver moving right would arrive in time but one moving left would not. Without a reliable way to tell their direction, I'm limiting diver attraction to columns 7 through 9, where a diver is close enough to be worth waiting for.

I'm now looking at how a diver in cols 0 to 3 should weigh against the bullets there, since a diver moving right would eventually reach col 8 and a bullet moving right would reach it sooner.

I'm settling the weights for the cols 4 and 5 case: a bullet should read as hot and an enemy as a target, but a diver there should stay near zero, and I'm checking whether one pair of weights can do all three.

I'm settling the cols 4 and 5 weights: a bullet there should read as hot, an enemy as a mild target, and a diver as a weak positive. I'll try a=−2 and b=3 and check whether tanh of the scaled sum separates those three levels cleanly.

I'm settling the per-column weights for the row signal: a danger cell in columns 0 to 3 adds +1 to the target side, while in columns 4 and 5 a bullet pushes negative and a diver pushes positive. For columns 6 to 9 the hot weight dominates, so an enemy there still reads as strongly hot even with the diver term added.

I'm weighing the cols 6 to 9 weights against the diver term, since with a hot weight of −4 and a diver bonus of +1.5, an enemy at col 7 still reads as hot even when a couple of targets sit in cols 0 to 3. The sum could land near zero in that case, so I need to check whether the row would stay neutral when it should be leaving.

I'm settling the hot-row weights at −6 for enemies in columns 6 through 9, with a small diver bonus of +1.5 in columns 7 through 9. A diver just beside the sub, or in column 9 moving left, should be picked up, while a friendly bullet in the row adds −1 each.

I'm settling the hot-row weights for the layer-one units: a friendly bullet in the row counts as −1, and I'm checking that the scale s=2 keeps neutral at 0 and separates the levels. Half-integer sums like a diver at +1.5 against a bullet at −1 land near ±0.5, which is still acceptable. Then I'll feed the row signals into the layer-two threshold X_y = tanh(k·(P_y + R_y − 1)).

I'm checking the inactive rows' layer-two values, which settle near −1 whatever the row signal is. For the active row, the value sits near zero when the signal is neutral, and it approaches ±1 as the row turns hot or targeted.

I'm working out how the diver channel and danger channel behave in the inactive rows, and noting that the sum over the seven inactive units comes out slightly off from −1. I'm treating row 0 as a special case, since there's no row unit for the surface, and considering whether a different design where inactive units output zero would avoid the error.

I'm settling how the surface row gets its signal: a layer-one unit R_0 that reads high when the sub is low on oxygen and has a diver to collect, and reads blocked otherwise. The up-row gates for rows 2 through 8 would then pull from the row below them, with row 1 looking up at that surface unit.

I'm settling the surface gate as a function of the two gauge bits: when the sub has a diver and low oxygen, the surface unit reads as a target, and otherwise it reads as blocked. I'll also add a global bias pushing the up logit when oxygen is low and a diver is in reach, while still letting hot rows block upward movement.

I'm making the surface-mode signal a global bias: when oxygen is low and a diver is in reach, the up logit gets a big push, while hot rows still block upward movement. Firing at targets on the way up is fine, but I'll pull the target weight down somewhat so the sub doesn't linger.

I'm working out the down-row gates the same way as the up gates: each row's down signal pulls from the row beneath it, and the bottom row can't move down, so it's treated as blocked. I'm also checking that the compensation constants still hold when the sub is at the surface or at row 8.

Now I'm setting the logit rules: when the sub is in a hot cell it has to leave, so it picks up or down depending on which move is open, and when it sits on a target it stays and fires.

I'm now setting the logit rules for the three actions: moving up, moving down, and firing. When the sub is in a hot cell, the up and down logits have to beat firing, so I'm checking that the hot-cell penalty β is large enough to force a move whenever a neighbouring row is clear.

I'm weighing the case where the cell above is hot: moving up would then cost a lot more than moving down, provided the row below is clear. If both neighbours are hot, it's unclear whether moving into one beats staying put, since the current row's danger might be a few frames away.

I'm setting the logit weights to α = 6, β = 6, γ = 6, and φ₀ = 2, and checking the hot-cell case first: with no neighbouring hazard, moving up or down scores 6 while firing scores −4, so the sub leaves the danger.

I'm now weighing the logit weights against the long-range case: with all neighbours neutral, the sub would drift randomly, so I need a term that pulls it toward a target or diver rather than leaving it to wander.

I'm settling the no-op logit now: I'll set it just below the fire logit, since firing at neutral rows wastes the cooldown, so fire only pays off when the sub is over a target.

I'm checking whether firing at enemies in columns 6 and 7 of our own row is worth it, since those count as hot and the sub should leave rather than linger.

I'm thinking about the long-range case now: when no target sits in the current or adjacent rows, the sub should drift toward targets further away instead of wandering. That needs gated units per row for targets above and below beyond the adjacent ones, and I have 15 layer-one and 9 layer-two units left to spend on it, plus a surface-navigation unit.

I'm sketching long-range targeting as a set of layer-one units, one per player row, each combining a row-distance term with the enemy presence in the columns that aren't yet hot, so a positive value means the sub should head up and a negative one means down.

I'm working out the long-range pull: a score for each row that goes positive for targets above and negative for targets below, passed through a tanh unit. Then I'd add a term to the up and down logits. I'm checking whether a weight of about 4 is enough to make the sub move rather than fire, and it looks marginal, so I'm trying 5.

I'm checking the long-range term at its extreme: with no targets anywhere the score is zero, so the up and down logits both sit at zero, the no-op logit at 2.5 and fire at 2, which means the sub holds rather than drifting.

I'm thinking about adding a mild center bias to the long-range units so the sub settles near the middle rows when nothing is in view. For the surface mode, I'm checking that a surface-mode weight of about 5 still lets the up logit beat hold when the cell above is clear, while a hot cell above still blocks it.

I'm checking the surface-mode term: a negative push on the down and fire logits is needed, but the long-range pull could cancel the up push, so I'm considering whether to raise the hot-cell penalty α to 8 so the hazard logic dominates.

I'm checking the surface-mode condition: it only holds when oxygen is below 60 and a diver is in reach, so the hot-row blocking still applies when oxygen is at 60 or above.

I'm checking the surface-mode case: if the sub is in row 1 with a hot cell above, the up move is blocked, so it goes down. But if row 2 is also hot, I'm asking whether moving up to surface would be a safer escape despite the cost of a diver, which means maybe relaxing the blocking when oxygen is high enough.

I'm checking whether the oxygen condition can tell when the sub is actually near the surface with a diver aboard. At oxygen 180 or higher the diver bar fills every column, so a live diver can't be seen there. Between 160 and 180, column 8 does show it, so I'm narrowing the surfacing-safe check to g8 with g7 unlit.

I'm checking whether the diver bar can be read in a way that tells surfacing-safe states apart, since column 7 lights up at two divers too, so that check won't work on its own.

I'm working out how to tell a diver-filled gauge from an oxygen-filled one, since at high oxygen the two look identical and the surfacing-safe check depends on that difference.

I'm settling the surface-mode check: the desired state is when the gauge shows the bottom gauge column lit with the second column off, which means a diver is aboard and oxygen is low. Anywhere else, the up move stays blocked, and I'm accepting the rare case of double-hot at row 1 as a trade-off.

I'm working out the row-0 navigation: once the sub surfaces it sits at (0,8) facing left, so it should move down only if row 1 isn't hot near column 8, and otherwise hold. I'm also setting up the position-tracking units for that row.

I'm working out the row-0 navigation: once the sub surfaces at (0,8) facing left, its back cell sits at column 9, so that trail is the marker for the start of the row. From the initial position at column 5 I need to move right until I reach column 9, turn back left to column 8, and then dive.

I'm settling the row-0 navigation for the start: the sub has to move right until it reaches column 9, then turn back left to column 8 before diving. I'm sketching a unit that flags being on row 0 without sitting at (0,8) facing left or at (0,9), which would drive the right move.

I'm settling the row-0 navigation with three unit groups: one that flags being in row 0, one for the sub sitting at (0,8) facing left, and one for it sitting at (0,9) facing right. The right-move logit then fires only when we're in row 0 and not already at either of those spots, while the left logit stays at a constant weight.

I'm now setting up the row-0 units for the start: one flagging the sub being in row 0, one for sitting at (0,9) facing right, and one for the sub at (0,8) facing left, which together drive the right, left, and down logits. I also need to make sure the in-water logic is muted on row 0, so the hazard terms don't push the sub around while it's on the top row.

I'm now setting the row-0 logic to dominate the in-water terms: with the hazard and position signals all at -1 on row 0, the up and down logits stay low, so I need the chosen row-0 action to carry a large positive weight. I'm also considering whether to fold a row-0 indicator into the sums so those in-water features go neutral on the top row.

I'm laying out the row-0 logic for the start: a unit that flags being on the top row, plus ones for sitting at (0,8) facing left and at (0,9), which drive the right, left, and down logits. I also want the right-move logit strongly negative once the sub is at (0,8), so it doesn't wander off the surfacing spot, and I'm planning to make the in-water terms neutral on row 0.

I'm checking whether a column-6-to-9 hazard check is enough to make a vertical move safe. Moving into row r at column 8 could be fatal from an enemy already there, or from one at column 7 or 9 that slides into 8 on the same step, or from a bullet crossing in.

I'm checking whether bullets sitting at columns 4 or 5 need any weight, and I'm asking what happens when an enemy bullet is fired from column 9 and moves left toward column 8.

I'm checking whether bullets fired from subs to the right of us could ever reach column 8 in time, and it looks like they can't: the earliest a sub fires from column 9 is after it has already moved to column 8. That leaves only a faster sub-speed setting as a case to rule out.

I'm checking that enemies at column 6 count as a hazard, since a sub there can fire a bullet into column 7 and then column 8 on the next step, so cols 6 through 9 all need to be watched.

I'm checking how divers at columns 4 and 5 feed into the target logic, since their weight makes the sub fire and hold rather than move. I'm also asking whether divers should be pursued more actively, because each one refills oxygen and is needed to survive past 200 steps.

I'm now looking at how divers near column 8 can be captured: each one passes through that column once, so tilting the vertical moves toward adjacent rows gives a reasonable chance of picking them up.

Now I'm working out how the diver supply holds up as the difficulty ramps: each surfacing speeds up spawns and movement, so the capture rate has to keep pace.

I'm checking how the spawn timing and movement speed change across surfacings, and noting that after about six surfacings enemies move every three frames, so the hot-zone check on columns 6 to 9 still gives enough warning to leave a cell.

I'm checking whether enemies that spawn at column 9 and move left could catch us in our row, since they can reach column 8 within a few steps at higher speeds, so I'd need to leave right after they appear. I also want to handle an enemy approaching from the left in our row by shooting it.

I'm checking how enemies in the target row are counted: those at columns 0 to 3 add to the target signal, and those at columns 4 and 5 add more through the second channel. Friendly bullets in the row subtract from it, so once one is fired the signal drops and I'd move on or hold. A hot enemy in that row still outweighs the target signal, so danger takes priority.

I'm working out the long-range units for the policy: one per player row, each pulling in the target channel from the rows above or below it, with the weight depending on which row it watches.

I'm setting up the long-range units now, with one per player row that counts enemies in the target channel from rows at least two away, weighting above and below differently. I'm also gating each unit so it stays strongly negative when the sub isn't in that row, and adding a small center bias that favors moving up for the lower rows and down for the upper ones.

I'm weighing whether distance should scale the long-range pull, since nearer targets matter more. A plain ±1 would leave ties stuck at hold, so I'm considering a distance-based weight that falls off as the row gets farther away.

I'm now writing out the long-range layer, weighting each row's enemy and bullet counts by its distance from the sub's row, and starting the second-layer combination.

I'm now tallying the layer-1 units: eight gated units, one per player row, to tell which row the sub is in for the later gating, plus a row-0 surface indicator.

I'm tallying the unit budget for the layer-1 and layer-2 counts: 28 units in each layer with 4 spare, and I still need to check how the starting position at column 5 on row 0 lets navigation reach the surfacing spot.

I'm checking whether enemies spawning at column 9 in the row I'm about to enter can reach column 8 before I can react. With a timer of 5 they don't move until several steps later, so I'd see them and leave in time. At higher ramp speeds the timer shortens, and I'm working through how many steps that leaves.

I'm checking the sticky-action case: if the previous action was a move, repeating it still carries the sub out of row r, but if it was a stay, the 10% repeat leaves us exposed for one more step. I'm weighing whether a fish approaching from the left in our row is a separate threat I still need to cover.

I'm now assigning indices to the layer-1 units: the eight per-row gate units first, then the nine row units, the eight long-range units, and three row-0 units.

Now I'm assigning the per-row gate units and the row-occupancy units for the layer-1 weights, with the pre-activation scaled so the danger columns push the signal strongly negative.

I'm settling the friendly-bullet and enemy weights for the row-occupancy units: danger columns get strongly negative values, divers in the middle columns count toward the target signal, and bullets in the row subtract from it.

I'm checking the sign of the friendly-bullet and enemy weights in the row-occupancy unit at columns 4 and 5, where a bullet would push the signal to about -2 and mark the row hot.

I'm checking the bullet case at columns 4 and 5: a bullet there pushes the signal to about -2 and marks the row hot. That's a bit over-cautious, but it's the safer side, so I'll keep it.

I'm settling the raw contributions for the row-occupancy units: cols 0 to 3 count +1 for an enemy, cols 4 and 5 give an enemy +1 and a bullet −1 while divers add +2, and cols 6 to 9 are strongly negative for enemies and bullets.

I'm now setting the friendly-bullet and enemy weights for the row-occupancy units at columns 6 and 7, so that a diver arriving there keeps the row marked as a target instead of dropping to neutral.

I'm weighing whether to drop the diver attraction at columns 4 and 5, since a diver moving away would keep the row marked as a target for no gain. Trying to infer direction from trails looks too complicated, so I'm considering a smaller positive weight for divers there that still makes bullets negative.

I'm weighing whether to treat bullets in columns 4 and 5 as threats too, since they'd reach column 6 within a step and leave only a one-step margin. Sticky repeats could make that margin risky, roughly a 10% chance of being stuck when the previous action was a stay, so I'm leaning toward counting them as targets.

I'm settling the bullet cases at columns 4 and 5 as hot, since an enemy bullet there would reach the sub's column within a step or two. Bullets further left, in columns 0 through 3, I'll still count as targets, because they point to a sub that can be shot from this row.

I'm settling the bullet case at columns 0 through 3 as a target rather than a threat, since a bullet that far left needs several steps to reach column 7. That leaves enough room to enter the row, fire once, and leave before it arrives.

Still writing the layer-1 weights for the long-range units...

I'm settling the long-range units now, weighting each row's enemy and bullet counts by distance and adding a small per-row centering bias so that, with no targets in view, the sub drifts toward the middle rows.

I'm settling the center bias for the long-range units now, leaning toward a small preference for the upper and lower rows, and I'm reconsidering whether a bias of that size would make the sub jitter between moves.

I'm setting the center bias so the sub prefers rows 7 and 8 slightly over rows 1 and 2, nudging it away from the edges where escape is limited.

I'm settling the row-0 units next, starting with the surface unit A0 and its gauge weights.

Still setting the row-0 units: the surface unit gets a weight of 4 on each row-0 cell with a bias of −2, and the trail unit is wired to the cell at column 9.

I'm now setting the layer-2 units that combine each row's pre-activation with its gate: the active row's signal passes through near zero when nothing is there, and flips to strongly positive or negative when a target or threat is present.

Still writing the layer-2 weights for the up-moves: each Up unit combines the row above's pre-activation with its gate, and I'm setting the weights and bias so it fires only when that row's signal is active.

Still writing the layer-2 weights: the Down units mirror the Up units, each pulling from the row below, and the LR2 unit sums the long-range activations with a large positive bias.

Setting the row-0 unit RIGHT0 next: it should fire only when the surface cell is active and neither the trail nor the facing cue marks that row, so I'm weighting the A0 signal against the B0 and C0 units with a bias that keeps it near ±1.

Still writing the layer-2 weights for the down-moves: DOWN0 should fire only when the trail marks the row below as clear, so I'm weighting B0 against the row-1 signal.

I'm checking the down-move unit DOWN0: it should fire only when the bottom-left trail marks row 0 as active and the row below isn't hot. With a ±1 margin of 0.5 and a gain of 4, both cases land near ±0.96, which gives a clean switch.

Now I'm setting the output logits, working out how to weight the up, down and no-op moves from the summed activations of the layer-2 units.

Now I'm working out the output logits: the no-op baseline shifts with the hot-row signal, and the fire and sideways moves get their own constants.

Now I'm working through the cases for the output logits with the no-op baseline at 2.5 and the hot-row term at −3. When a row above is hot and the long-range signal is strong, the up logit can still beat the other moves, so I need the hot-row weight to dominate the long-range term.

I'm weighing whether the long-range term should push the sub up or down in hot-escape cases, since at this point its influence could outweigh the escape signal. I'm considering gating it so it has little effect when a row is already hot, or simply accepting a small chance of drifting toward danger.

Now I'm weighing the up and down logits with a larger penalty on hot rows, since a block in the row above should dominate any long-range pull.

I'm making the attraction toward a target neighbor strong, around 12, so the sub commits to it, but I'm keeping surface mode large enough that a diver below can't pull the sub off the surface when it needs air.

I'm tuning the attraction weights so that a hot row above strongly suppresses upward moves, while a target neighbor still pulls the sub toward it. I'm also weighing whether making targets weaker than hot blocks would be cleaner, since the current scaling leaves them roughly equal in strength.

I'm checking whether the asymmetric thresholds on the Up and Down units can separate hot rows from targets, since with a plain tanh both end up at ±1. An offset in the pre-activation might keep inactive units saturated while letting a target give a weaker positive signal.

I'm tuning the Up and Down logit weights so the hot-row penalty stays large, and I'm trying to keep the long-range pull small enough that it can't push the sub into a blocked row.

I'm settling the logits for the no-op, up and fire moves, with the surface-mode term pushing up and the target and long-range terms weighing in. With no target or hot row present, the up logit only gets a small nudge from the long-range signal, so it sits near the no-op baseline and the sub drifts slowly. The check on the fire move still holds: when a target is directly above, up stays below fire.

Weighing the long-range term against the hot-row penalty, I'm seeing that with δ at 5 and a hot row above, the up logit still ties the down logit at 5 when the sub is in row 0 with nothing below, so the long-range pull can drive it into the blocked row. I need to gate that term off whenever a row is hot.

I'm trying a gate that keeps the long-range pull from acting when a row above is hot: an AND-style unit that only fires for a positive long-range signal when nothing is in the way. Then the up and down logits can take that gated term instead of the raw one.

Gating the long-range pull off whenever a row is hot doesn't work with layer 1 alone, since the current row's hotness is a layer-2 feature. I'm considering whether to give each row two layer-1 units, or to gate on the up and down blocking in layer 2 instead, and both options look like too much machinery.

I'm checking the case where the row is hot: the up-versus-down gap is about 10 from the blocking term, so the long-range pull has to stay under roughly 2.5 to avoid flipping it. That leaves the drive toward the target too weak at the baseline, so I'm lowering the no-op baseline to see if the sub can still move.

I'm still going over the trade-off between the hot-row penalty and the long-range pull, and I'm considering raising the hot penalty to about 14 so I can afford a stronger long-range weight without an upward move into a blocked row.

I'm settling on a larger hot-row penalty of 14 with the long-range weight around 4.5, since that keeps the sub from drifting into a blocked row. Then the fire move at a target directly above needs a strong enough gamma to beat the up logit even in surface mode.

I'm checking the surface-mode weight M = 7 against the hot-row case: with no hot row above, the up logit comes out at 7 plus the long-range term, which should still clear the no-op baseline.

I'm trying to make the long-range pull work in surface mode too: the sub should drift up when it needs air, so I'm adding a term that tilts the long-range units toward up based on the surface signal, and that way the no-op and fire baselines can stay low.

I'm checking whether the surface-mode push can come from the long-range units without the sign flipping on the mid-oxygen case, since the simple linear sum gives the wrong direction there.

I'm checking whether the surface-mode push can come from the gauge bits without flipping sign in the mid-oxygen case, since a plain linear combination of them doesn't separate the surface condition cleanly.

I'm checking the surface-mode case with the hot row directly above, and I'm working through the numbers for when the sub is in row 0 with nothing below. I'm trying a larger surface weight of 10 so that up wins by sheer magnitude, while the long-range term and the hot-row penalty stay in their current ranges.

I'm checking whether the surface-mode push can work by adding a constant to the row units, but that makes neutral rows above look like targets, which would pull the sub into hot rows as well.

I'm working out how to add a surface-mode push to the up unit through layer 2, using a term that is strong when the surface signal is active and close to zero otherwise. For a neutral row above, that makes the up unit look target-like when the sub needs air, while a hot row above still gets a large negative pre-activation and stays blocked.

I'm checking the surface-mode push through the layer-2 up unit: a surface-active term gives a neutral row above a target-like pre-activation, but the same term must not flip a hot row above to neutral. The inactive case also drifts off its ideal value of about −1, which adds a small error I'll need to account for.

I'm making the gate strong, with a large factor on the active-versus-inactive term, so the inactive units sit firmly near −1. The payload for an active row then reduces to a linear sum of the row-above state and a surface-mode constant, and I'm checking that constant stays small enough that a hot row above still reads as blocked.

I'm working on the surface-mode push for the up unit: adding a term scaled to the surface signal so a neutral row above reads as attractive, while a hot row above still stays blocked. I'm checking the leftover long-range pull, which could still drag the sub downward in that mode.

I'm checking the surface-mode case with the sub in row 0, where a positive surface signal and a 0.91 "up" activation should still win over the long-range pull, and the blocked case with a hot row above still lands well below the no-op baseline. For row 1, the surface unit's contribution flips the sign of the row-above payload, so a hot row above reads as blocked and a neutral one reads as attractive. I'm wondering whether the surface-mode suppression of fire and no-op is still needed on top of this.

I'm now considering whether the down units also need a surface-mode adjustment, since a target below could otherwise pull the sub downward while it needs air. I'm working through a term that makes a neutral row below look hot in that mode, so the sub only goes down when it's escaping a blocked row above.

I'm checking whether a suppressed down move can still win once the long-range pull is included, since at x=−1 with the hot row above the long-range term swings the up and down scores enough to flip the choice.

I'm now trying to gate the long-range pull by the current row's state, since it should only matter when the sub is at x = 0. A single tanh can't cleanly switch it off in a hot row, so I'm considering using the long-range term only as a small tie-breaker and handling seeking some other way, perhaps by spreading target signal from neighboring rows into each row's state.

I'm now planning to split the row-state units into two variants, one that tracks targets above a row and one that tracks targets below, so the up and down moves each see only the direction they need. That would mean roughly doubling the row units in the first layer, and I'm checking whether the layer-1 budget still fits.

I'm considering a row-specific split for the up and down signals, where each row tracks targets above or below separately so the directional moves stop interfering. The catch is that the own-row state used for firing must stay clean, so I'm weighing whether the extra units fit the layer-1 budget.

I'm now counting the layer-1 units needed for the surface and long-range variants, and the total comes to 34, which is over the 32-unit budget. I'm considering whether the own-row state can be dropped by building the firing signal from the up and down variants in layer 2, but a row that is neutral leaves that combination ambiguous.

I'm thinking about the row-0 navigation next, maybe folding the right-moving check into a single first-layer unit instead of several separate ones, so the layer-1 budget stays under 32.

I'm checking the right-moving check for row 0: the sub is there and the cell at column 9 is empty, so a single first-layer unit can cover it with a threshold. The left-moving case is simpler, just column 9 occupied and not by a sub facing right.

I'm trying to merge the left-moving and down-moving checks for row 0 into one first-layer unit, which would get the layer-1 count back under budget.

I'm now tallying the layer-1 budget again: the 8 position units, 8 own-row units, 8 up-variant units and 7 down-variant units already come to 31, so the two row-0 units push it to 33, one over. I'm checking whether the surface row's up-variant can be dropped, since it's only needed for row 1's payload.

I'm now considering whether the gauge term can be folded into the row-1 position unit, but adding it there would break that unit's gating. Instead I'm noticing that row 1 has no rows above it, so its up-variant is just its own state, which would let me drop a separate own-row unit.

I'm recounting the layer-1 units: the 8 position units, the own-row units for rows 2 through 7, the up-variants for rows 0 through 7 with row 1 reusing its own state, and the down-variants for rows 2 through 8 with row 8 reusing its own state, plus two row-0 units. That totals 31, leaving one spare. Layer 2 would then need about 26 units, so there's room to spare there.

I'm now weighing whether the spillover into each row's up-variant should count only targets, or also the nearby diver threats, since a diver drifting toward the sub's column can arrive within a few frames.

I'm now weighing whether diver threats in the far rows should add to the spillover, since a diver at the right columns can reach the sub's column within a few frames. I'm inclined to include diver contributions with a modest weight, and I'm checking that enemies, which register on both channels, cancel out so only divers push the signal positive and bullets push it negative. The spillover still has to stay smaller than the hot-row penalty.

I'm settling on the spillover part: the upward and downward variants will carry target signal from distant rows, so the separate long-range units can go. That brings the layer-1 count down, and I'm checking that the hot-row penalty still outweighs the spillover. For the current row, firing depends only on its own state, and movement at column 0 picks up or down based on whether the neighboring row is clear or has a target.

I'm weighing a small per-row bias for idle movement, nudging rows near the bottom to prefer moving up and rows near the top to prefer moving down. The concern is that the layer-2 scaling would amplify even a modest bias into a strong push, so the payload has to be tuned carefully.

I'm working out whether the bias idea holds up: without a cross-row signal, idle movement at the edge column would just hold still, and the spillover into each up and down variant should cover the rest. I'm also checking how the two directional signals interact when both are positive, since moving up can leave the row below still reporting targets and set off a back-and-forth.

I'm working out how strongly the spillover from far rows should saturate the up and down signals. A small raw spillover would get pushed to nearly full strength by the layer-2 scaling, so I'm considering a smaller gain on those units, though that would weaken the hot-row penalty.

I'm weighing the spillover from far rows as graded values, so that nearer targets push harder than distant ones. I'm setting the hot-row penalty larger than the spillover, and I'm starting to write the combined logit for the up direction from the weighted activation, the position penalty, and the spillover term.

I'm weighing a fire-or-escape payoff at x=+1 where the sub's own row carries a graded target signal, and I'm checking whether firing while a diver sits in the row should still count as staying put.

I'm checking whether a diver directly in the sub's row, with the sub at column 8, gets picked up on the next frame, and I'm noting that the bullet's negative signal still leaves the net positive.

I'm checking whether a diver in a far row is actually a threat: one at column 7 moving right would leave a trail at column 6, and one moving left would leave its trail at column 8. Enemies and the sub's own wake also show up in those trail channels, so I need to separate them before letting the signal push the sub around.

I'm now checking whether a diver in a far row is a threat by looking at its trail cell: a diver at column 6 moving left would leave its trail at column 7, so I'd subtract that trail's signal to avoid chasing divers that are moving away from the sub.

I'm settling the trail handling for divers: I'd count divers in columns 6 through 9 as positive signal, but subtract the trail channel where a diver heading away would leave its wake, so a diver moving off the sub's column doesn't pull the sub toward it.

I'm now settling how the diver trail cells should cancel: a left-moving diver at column 7 would leave its trail at column 8, so that trail should subtract from the pull, while my own sub's back cell in its row doesn't get that weight.

I'm checking how diver attraction from far rows should feed into the spillover, since a diver at columns 7 to 9 can reach the sub's column within a few frames and stay there for several more.

I'm now checking how far-row divers feed into the spillover, since their pull should fade with distance, and I'm weighing whether the spillover scaling will keep the signal graded instead of flattening it into saturation.

I'm now weighing a graded spillover, where a target at distance one counts as a raw +1, distance two as +0.5, and so on down, so nearer threats pull harder than far ones. I'm also tuning the scaling so the layer-2 units don't saturate every distant target to the same value.

I'm now plugging the spillover values through the activation scaling: the own-row target lands near 0.88 at the Up unit, the distance-two target near 0.74, and so on down to about 0.49 at distance four. The hot-row penalty saturates at -0.9. Multiplying by the logit scale gives 12.3 for the own target, 10.4 at distance two, 8.7 at distance three, 6.9 at distance four, and -12.7 for the hot row.

I'm checking the threshold math for the spillover: with the own-row target at 12.3 and the distance-two target at 10.4, a graded pull still separates the nearer threat from the farther one. Two far targets at distance three sum to a similar strength as one at distance two, so the policy stays roughly even between them. In the hot-escape case, moving up gives about 1.3 and moving down gives 14, so the sub drops away from the enemies, while the noop and fire values sit much lower.

I'm checking the fire-versus-move balance at x=+1: fire comes out near 9.9 while the up move stays negative, so firing wins by a wide margin. I'm also considering shrinking the no-op term to 2.5 + x so it doesn't outweigh firing when a target is directly ahead.

Now I'm settling the surface-mode case, where the up neighbor should still look attractive even when it's neutral, and chasing a target below should pull the sub downward.

I'm trying to keep the surface-mode bonus for the up move from flipping a hot cell to neutral, since a raw of -4.5 saturates to -1 and adding the bonus would cancel it out.

I'm settling the Up payload as 3R plus a 1.5 surface bonus, so a hot cell stays negative even with the bonus added. That keeps the neutral up cell attractive in surface mode, but it flattens the graded spillover a bit, since distant targets saturate closer to the same value.

I'm now working out the surface-mode case by folding it into the row signals themselves: in surface mode, each upward row gets a small extra pull, so the neutral cells above stay attractive. I'm checking that a bullet sitting at columns 4 to 5, with its raw value near -1, doesn't end up looking neutral once that pull is added.

I'm checking whether a one-hot style signal can flag only the surface-mode case, since a linear combination of the two gate inputs can't produce the pattern I want, which makes the surface bonus harder to fold into the row signals.

I'm now checking whether a gauge column can separate the case where the sub is at surface-adjacent oxygen with no divers from the others, since the current inputs can't isolate that pattern on their own.

I'm working out how to encode the surface-mode pull with gauge columns, since the gate combinations don't separate cleanly with a single term, so I'm checking whether a combined term can cover the missing case.

I'm weighing a shared surface-mode unit, built from the two gauge inputs and passed through a tanh, that would switch an extra upward bonus on or off. I'm planning to add that bonus to the Up payload so the neutral cell above stays attractive in surface mode without turning a hot cell neutral.

I'm setting the surface-mode term to c = 0.6 times (SMU + 1), so the Up payload gets 2c = 1.2 extra when surface mode is on. That keeps a neutral cell above looking attractive (about 0.83) while a hot cell stays negative (about -0.95), and a bullet at columns 4 to 5 still reads as negative (about -0.9). Down gets the mirror term, so in surface mode it drops to about -0.83 and loses some of its pull.

I'm now checking the down-direction penalty in surface mode, adding a fixed push away from the surface on the Down output. With the up cell hot, the escape check gives Up around 0.7 against Down at 9, so the sub moves down, not up. When a target sits below with the up cell clear, Up still comes out ahead, about 11.6 versus 8.9.

Working out the surface-mode bonus: a shared unit R_up_0 that equals the surface-mode signal, so the Up payload for row 1 becomes three times that signal plus a small surface term. In surface mode the neutral cell above stays attractive, while a hot cell stays negative.

I'm setting up the layer-2 gated units, with the gate pulling each one off when its row is hot, and I'm tallying the unit count: 27 used, leaving 5 spare.

Still working through the weight offsets: the gated layer-2 units get a bias that sits just under the active threshold, so when a row is hot the unit is pushed far negative. I'm setting the output-layer bias to cancel the sum over the X units, since each contributes its weight times roughly -7 when submerged at column 8.

I'm checking the row-0 case, where every in-water unit is inactive: the up and down logits both land at zero, while noop sits around 1.5 and fire is strongly negative.

I'm setting the row-0 overrides now: a right-move term that starts at -10 and jumps to 10 when RIGHT0 is on, a matching left term, and a down term that rises to 10 under DOWN0. That way the sub heads toward a target in the top row, and when row 1 is hot it still avoids going down.

I'm now considering a down-direction term tied to a row-0 indicator, so that when row 1 is hot the down logit gets pushed well below noop and the sub doesn't drift into the danger row.

I'm working out a row-0 down term: the down logit gets a penalty whenever the sub is in the top row, plus a boost only when the DOWN0 flag is on, so the sub won't drop into a hot row 1.

Setting up the first row-0 unit to detect whether the sub is in the top row, using a weighted count of lit cells there. For the third unit I'm working on a three-level gate at column 9 so the top-row case can be told apart from the rest.

I'm now defining a row-0 detector: a layer-1 unit that sums the occupancy of the top row's columns 0 to 8, so it can tell when the sub is in the top row and the right-hand side is clear.

I'm setting the row-0 detector to a bias of -2 so the top-row case saturates cleanly: when the sub sits in row 0 with both cells clear of column 9, the unit reads about 0.96, and when it isn't in the top row it reads about -0.96.

I'm checking whether the row-0 detector can be dropped by inferring "top row" from the all-inactive pattern of the in-water units, since the sub always sits at column 8 when submerged.

I'm using the input that gates row 0 to skip a separate top-row detector, since the layer-2 unit can infer row 0 from the fact that no in-water row unit is active. The margin is thin, so I'm checking that a steeper gain still saturates cleanly in both cases, and I'm wondering whether the sub might ever be off column 8 while submerged, which would break that inference.

Still working out the row-0 left term, using a gate built from the top-row detector.

I'm working out the down-direction gate for row 0: it should fire only when the back cell sits at column 9 and row 1 is clear, using a bias chosen so a hot row 1 pushes it firmly negative.

I'm checking whether the down-direction gate for row 0 can be built from the back-at-column-9 signal and the row-1 detector, with the row-1 term weighted so a clear row 1 gives a positive value and a hot one a negative value.

Now I'm checking whether the down-direction gate for row 0 can be built from the back-at-column-9 signal and the row-1 detector, with the threshold set so that a clear row 1 leaves the gate positive and a hot one flips it negative. In open water the column-9 term drops to zero, so the gate stays off there.

Now I'm setting the down logit to add a term for the row-0 flag and subtract one for being in the top row, aiming for a value near 8 when DOWN0 is on and I'm at the top.

Checking the row-0 down logit at the top row, with the gate off in open water so the penalty stays small there. Next I'm considering what happens at the start of the game, where the sub begins in row 0.

Checking the 6-diver cap now: if the count is already full, a diver that reaches the sub should pass by without being collected, so I'm wondering whether the rule for that case is actually written correctly.

Now I'm writing out the per-row contributions for the own-row units, weighting enemies, divers, bullets and the sub's neighboring cells by hand.

Checking whether the enemy trail sitting at column 7 in our own row, and the cell at column 8, can be told apart from the sub's own body when a diver overlaps.

Still weighting the own-row units: I'm giving the enemy trail at column 5 and 6 a small negative weight, and I'm setting the diver contribution at columns 4 and 5 so it lands as a modest positive.

I'm now weighing whether a diver at column 4 moving right, which would take up to 24 frames to reach the sub, is worth waiting for given how much each diver adds to oxygen.

I'm now sizing the spill term for rows above the target row, scaling each higher row's pull by its distance and counting divers and enemies there.

I'm now defining the spill term for the rows above: for each higher row, I'm counting enemies and divers in the low and high column ranges, with the bullet and enemy-trail penalties subtracted, and then weighting the total by how far that row sits from the target.

Now I'm working out the surface-mode term for row 0, where a tanh gate reads the oxygen and surface flags and comes out near -0.9 in most states and -1 in the worst one.

Now I'm working out the layer-2 units for the up direction, weighting the target row's pull and the spill from the row below, and checking whether a k2 of 2 keeps the saturation clean.

I'm working out the up-direction payoffs for each row: the target one row above gives about 0.95 once scaled, and the pull fades with distance, from roughly 0.85 down to 0.52 for the farther rows.

I'm checking the surface bonus for the up direction: with a gain of 2, a surface-mode bonus of about 0.66 times the surface flag term keeps the hot-row penalty from overwhelming the signal. When surfacing it lifts the neutral up value to roughly 0.83, and when not surfacing it adds nothing.

I'm settling on a hot-up value near -0.66 after tanh, which pushes the up logit to about -9.3 in escape mode, so down wins there with roughly 98.7% confidence. I'm trying a smaller surface-bonus coefficient of 0.5 instead, which gives neutral up around 0.76 and hot up around -0.76, a roughly symmetric split.

Now I'm setting the down-layer units to weight the row below with a factor of 2, and I'm noting that the surface-mode penalty on the down direction comes to about -5 when surfacing and nearly zero otherwise.

Next I'm setting the output logits for the layer-2 groups, summing each block of hidden units and adding offsets to get the x, a, and b values.

I'm now writing out the output logits from the summed hidden groups: noop, up, down, fire, left, and right each get a linear combination of the x, a, and b sums, with the down logit also picking up the surface penalty, the DOWN0 gate, and the IN0 term. Checking the water case, left and right both come out near -10, so they stay well below the others.

Now I'm checking the x=0 case with a set to 0.948 for the target directly above, where the up logit comes out near 13.3 and clearly beats the noop logit of 2.5.

Now I'm checking whether the far-target case holds up: a target five rows up gives an up logit near 7.3 against noop's 2.5, so moving up is strongly favored. I also want to confirm that when two targets, one above and one below, have similar strength, the policy settles toward one of them, since each step toward one raises its score and lowers the other.

I'm checking the hot-state cases now: when the up unit is hot and down is clear, down wins clearly, and with both neutral the two split evenly. At the bottom row there is no down unit, so down stays blocked there.

I'm checking the surface-mode case: with the up direction carrying a surface bonus and the down penalty raised to 8, a neutral target above still wins when we're at x=0, and the hot-escape case shifts toward down at about 94%.

I'm checking the surface-mode down penalty, weighing a value of 5 against the escape case: with it set there, the up move wins for a target above and escaping a hot row still favors down with high confidence.

Now I'm checking the surface-mode case, where the hot-escape up value and the down penalty need to stay balanced so that escaping a hot row still wins.

I'm checking whether a friendly bullet can make a row look hot, since a row with only my own bullet and no enemy would currently read as negative. That would push me to escape rows I'm actually shooting at, so I need the bullet term to not drive the row negative.

I'm weighing whether a friendly bullet should count against a row, since a row with only my own shot in it would currently read as hot and push me to escape from a target I'm actively shooting at. Removing the bullet penalty entirely would fix that, though the row would keep looking like a target until the shot lands. A small penalty doesn't help, because the steep tanh gate still tips a row with only my bullet into the hot range.

I'm settling on a friendly-bullet penalty of 0.4 instead of 0.5, so a row holding only my own shot lands in the neutral range rather than the hot one, while a target with my bullet still reads as positive.

I'm weighing whether a fixed penalty on friendly bullets can work, since the steep gate would read a bullet-only row as hot. Getting a clean dead zone around zero would need a second unit per row, which doubles the X units and isn't worth it for this policy.

I'm weighing whether to drop the friendly-bullet penalty altogether, since a row with only my own shot would otherwise read as hot. Without it I'd spend a few extra frames pressing fire on the same target until the bullet lands, which seems an acceptable cost.

I'm weighing a different fix for the friendly-bullet problem: a spare first-layer unit that detects a bullet in flight in columns 4 to 7, which roughly means the cooldown isn't ready yet. That would let the policy see when firing is blocked and prefer moving toward other targets rather than waiting in place.

I'm settling on the friendly-bullet handling: no bullet term goes into the row value at all, since any penalty would make a bullet-only row read as hot. The own-row contributions stay as enemy-side terms only, with the bullet effect coming from the cooldown rather than the row score.

I'm settling the row-value terms: the enemy-side contributions get scaled by 1.5 before the activation, and a row with a diver moving away and a target still counts as positive. I'm also checking that two enemies in the same row don't cancel the hot signal from one far-right enemy.

I'm re-checking the enemy-side terms and confirming that the trail cells in my own row don't come from my own submarine, so those contributions stay clean. Next I'm moving on to the spill contribution for the upward row.

I'm now working out the down-direction spill: each row at distance d contributes its target-ish value scaled by the weight for that distance, using the same enemy-side terms as before. Divers at columns 0 to 5 still count as neutral rather than targets, and bullets far to the right are treated as mildly negative.

I'm now setting the spill weights for rows at distances one through six, scaled by 1.5 before the activation. A few neutral targets spread across nearby rows would still push the upward signal to a strong positive value.

I'm checking the enemy-side contribution for a neutral target in the own row: a raw value of +0.5 from an enemy at columns 4 to 5 gives a moderate positive activation that still sits below the strong-target level.

I'm working out the bullet-in-flight case for the own row: a bullet at column 4 gives a neutral reading, and once it reaches column 6 the row turns hot, so I'd need to move out on that step to stay clear.

I'm checking the bullet-in-flight case for the friendly bullet in the own row: once the bullet reaches column 6 the row turns hot, but I'd still have about a two-step margin to move out, and a sticky repeat of fire or noop only stretches that margin slightly.

I'm checking how a sub at column 3 that fires could look in my own row: the bullet at column 4 nets to neutral with the sub's +1, and it only turns hot once it reaches column 6, which still leaves me about two steps to move out.

I'm checking the input-unit contribution for the own row: after a bullet is fired and its cell comes into the picture, the value collapses under a tanh pre-activation. I'm also working through how the output responds to bullets at columns 4 and 5 coming from a submarine on the right moving left.

I'm checking the row-zero case for the in-water feature: the sub's cells at column 8 and 9 mean the sum over rows is effectively fixed, so I need to see whether a constant small penalty on the down direction holds up or needs a bias to compensate.

I'm working out the down-direction value for row 0: the bottom-row term nets to +8 once the in-water penalty is applied, while noop, up, fire and right are each scored separately from their own unit contributions.

Checking the down-direction value at the start, the policy gives down a probability of roughly 0.998, since its score is about 8 against much smaller scores for the other actions.

I'm checking the down-direction value for the start position: the in-water term holds it near 8, while right scores about 9.9 once the sub is facing left at column 5. So right wins, and I'll keep pressing right as the sub moves toward column 8.

I'm checking the left move from the far-right cell: with the sub back at column 9 and the bullet-side terms balanced, left scores about 9.5, well above noop and up, so I'm pressing left and then checking the down value once I'm at column 8 facing left.

I'm checking the down-direction value on the step after that: with the sub now at column 8 facing left, the upper-row term keeps down positive even at a small negative raw reading, so the down score stays ahead of noop.

I'm checking the early-game timing: enemies start spawning around step 20 and divers around step 30, so I should be able to get down quickly. Moving down from row 0 with no air left is fine since the surface flag starts True, but returning to row 0 later would be fatal, so Up from row 1 has to stay blocked unless I have oxygen to spare.

I'm noticing a sticky-action problem: if I move up into row 1, there's a 10% chance the environment repeats "up" on the next step, which would surface me with no oxygen and kill me. So I should avoid stepping up into row 1 unless I really need to.

I'm still working out that row 1 should be avoided as a destination from row 2, since a sticky "up" would surface me, and it's only acceptable in surface mode. Any enemies I want to target there will have to be weighed against that 10% risk.

I'm weighing whether to ever move up into row 1 from row 2, since a sticky "up" there would surface me and kill me if I have no divers. Going for an enemy in row 1 isn't worth a 10% death risk, so I'd only do it in surface mode.

I'm now adding a penalty to the upward move from row 2 into row 1 when not in surface mode, since a sticky "up" would surface me early. I'm making the offset depend on whether surface mode is active.

I'm adjusting the upward move from row 2 into row 1 so it's heavily penalized outside surface mode, since a sticky "up" there would be fatal. With the surface term included, that move comes out blocked in the non-surface case, so if the down move is also weak, the two are nearly tied.

I'm weighing whether row 1 is worth entering at all: from row 2, a sticky "up" would surface me, so I'm keeping that move heavily penalized outside surface mode, which leaves noop ahead of it.

I'm settling on keeping the up-move from row 2 into row 1 blocked outside surface mode, since a sticky repeat there would be fatal, and I'll leave row 1 mostly unused.

I'm working out the upward-move values for the rows above: rows 3 to 8 get a payload that adds a surface bonus, which comes out near +1 in surface mode and near zero otherwise. For row 1, the same structure applies but the bonus is scaled differently, so surface mode gives roughly +0.99 and non-surface roughly −0.95.

Now I'm checking the sticky-action risk when moving up or down into a neutral row: a repeated action could carry me one row further into a hot row, so I need to weigh that 10% chance.

I'm now considering a hot-spill term: a neutral row that has a hazard two rows away can still be dangerous, since a sticky repeat of the same move could carry me into it. So I'd penalize the upward value using the cells in the row beyond it, weighting enemies and bullets there at about -1 and checking how far that shifts the scores.

I'm weighing how a sticky repeat could carry me into a row two away that's currently hot, and whether a soft penalty on that spill is worth the mobility it costs, since the tanh slope makes even a small raw penalty block the move. I estimate the lethal-entry risk at roughly half a percent per move, which I'm still comparing against the cost of blocking movement too often.

I'm weighing a small spill penalty for hot cells two rows away, since a sticky repeat could carry me into them. The tanh's steep slope near zero means even a small raw penalty would block the move, so I'm considering a gentler scaling of the raw value.

I'm checking whether the sticky-action risk into row 1 and the other rows is worth fixing with a spill penalty, and I'm leaning toward accepting it since the tanh slope would block movement too often. Then I'll look at how divers in the neighboring rows pull the up and down values.

I'm double-checking the oxygen thresholds: the second light is lit only at 60 or more, so surface mode means oxygen under 60 with at least one diver, which makes the eighth light depend solely on the diver count. I'm also checking what happens when oxygen hits zero, since the episode ends there.

I'm checking how the oxygen runs down: the sub loses one unit per step while it's below the surface, and the episode ends once oxygen hits zero at the next step's check. I'm also reconsidering the spawn rate, since the spawn timer drops with each surfacing, so later in a long game enemies could show up every couple of steps and give more targets.

I'm checking how enemies entering from the right would move through rows that aren't mine, since a row with a hazard at columns 6 to 9 stays dangerous for a while.

I'm checking how a diver moving right would leave a trail behind it, and whether an enemy's trail at column 6 matters for the row's danger score.

I'm checking how divers at columns 4 and 5 would be scored: a diver there is attractive, but I'd need to subtract the trail, and an enemy moving left would look hot even though it's harmless.

I'm weighing whether to narrow the diver attraction in my own row to columns 7 to 9 only, since a diver at column 4 or 5 could take many frames to reach me. Counting only the far-right cells would keep me responsive once a diver actually arrives, but then I'd miss the chance to wait for divers that are already close.

I'm settling the diver attraction for the cells in my own row: columns 4 to 5 stay weighted so enemies there come out positive, with the bullet penalty offset by the diver bonus, while column 6 carries a heavy penalty for enemies and bullets that outweighs the diver term.

I'm checking whether a diver at column 8 moving right, with its trail at column 7, nets out to a positive value in my own row.

I'm settling the spill question for now: a diver two cells ahead that a sticky move could reach only adds a small pull, and at roughly one-in-a-hundred per step the risk of wandering into the water's lethal edge is low enough to accept. I'll keep checking how often the sub might drift out of column 8 if the left and right moves ever stop being suppressed.

I'm checking the case where the sub drifts to column 9 while facing right: the back cell stays on column 8 so the policy still treats the row as active, but my shots would head the wrong way and enemies could appear right beside me. That looks rare, so I'll accept it for now.

I'm making the left and right logits much more negative, with a base of −14, so that only a clear +10 offset from the facing check can activate them. I'm also checking what a sticky repeat of left does after I've moved left from column 9, since it could carry me to column 7 facing left and then have me reverse back toward column 8.

I'm settling the surfacing step: once the sub moves up into row 0, the refill happens and the next check is whether row 1 is safe, so a down move follows. I'm also making sure the surface flag stays set while I sit there, which it does.

I'm checking the row-to-row lookup terms: row 1 pulls from its own up-term, which feeds the up move into row 2, and the bottom row's down-term feeds the move into row 7. The middle rows each use three terms, one for staying put and one for each vertical move, and I'm confirming the index ranges line up with the up and down actions.

Still assigning the layer-1 unit indices, checking that the down-action lookups land on the right units.

Still mapping out the weight indices for the row-based units, working through the strided column and channel offsets for each row.

Still laying out the strided index ranges for the row-based units, checking each column group against the channel offsets.

I'm listing the weight contributions for the own-row cells, with the diver and enemy terms scaled by 1.5, and now adding the spill weights from neighboring rows, with the raw values falling off by distance.

Still working out the spill weights for each distance, deciding whether to leave divers in columns 4 to 6 out of the spill terms. Then I'll build the upward-row unit as the own-row term plus the spill from each row above it, weighted by distance.

Still counting the spill weights for the R units: the down-side ones sum the contributions from rows below, with distance-scaled terms, and I'm tallying how many statements that adds up to for each group.

I'm working out how to write the weight matrix compactly, since each recurrent unit draws on the same cells in its own row and on spill from rows above and below. Filling entries as 2-D blocks, with strided row ranges and shared column ranges, looks like it will cut the number of statements a lot compared with writing every weight out individually.

I'm weighing whether to write each spill term as its own assignment per unit or to batch the same input across a contiguous block of R_up units in one 2-D fill, since the per-unit version gets very long.

Still checking the spill layout, and this time I'm weighing whether to fold the diver and enemy columns into one explicit value list per unit instead of separate statements.

I'm thinking about whether to trim the spill terms down to just the target channel in columns 0 to 5 and the diver channel in columns 7 to 9, since counting enemies in distant rows as divers would skew the weights.

Still counting how many statements the spill terms need, since each row pair takes three lines. Then I'm weighing whether grouping the own-row contributions would cut the total from about 180 lines.

Now I'm weighing how to reorder the recurrent units so the three per-row units are evenly spaced, which would let the own-row statements share one stride across units.

I'm renumbering the recurrent units so each row's three units sit exactly seven apart, which lets one strided fill cover all of a row's own-row contributions.

Still mapping the unit indices: I'm trying a layout where each row's three recurrent units are spaced seven apart, so one strided fill can cover that row's own-row terms, but the spacing collides with the unused slots at the edges, so I'm checking which units are actually free.

I'm mapping the recurrent units onto evenly spaced indices so each row's three units sit seven apart, which lets one strided fill set all of that row's own-row weights at once. Rows 2 through 7 fit cleanly, while the edge rows leave one slot doubled up on unit 16, which I'll keep unused.

I'm tallying the own-row contributions at 9 statements per row, which makes 72 in total across the eight rows, and I'm working out how the spill terms fit into the same layout.

Still weighing the spill terms: for each pair of rows, the influence on the upward-row units falls off with distance, so I'm deciding between one 2-D block per source row or a separate statement per target unit, which is more verbose but easier to check.

I'm weighing whether to write the spill terms as one 2-D block per source row, which cuts the statement count, or as separate per-unit statements, which are longer but easier to check for shape mistakes. I'm leaning toward the per-unit version for clarity, even though it means many more lines.

I'm going to drop the diver and extra channel terms from the spill and keep only the ch1 target cells in cols 0 to 5, which cuts it to one statement per row pair. Divers in distant rows don't matter much for the policy, since their pull only mattered when they were in the adjacent row.

Still weighing the spill, I'm settling on two statements per row pair: one for the target channel in columns 0 to 5, and one explicit 14-value list covering the diver and enemy columns 7 to 9.

Still working through the diver and enemy spill for distant rows, and I'm checking whether far-row enemies at columns 7 to 9 cancel out properly so they don't pull the policy toward them. Then I'll confirm the danger penalty of −9 per cell in columns 6 to 9 for the own-row units.

Still writing the layer-2 weights: each output unit gets its own diagonal link plus the recurrent unit that drives it, along with a bias.

Now I'm setting the layer-2 weights for the up-row and down-row units, working through each unit's index and the bias that goes with it.

Still writing the layer-2 weights: each down unit gets its diagonal link plus a weight from the recurrent unit below it, along with the matching bias.

Still writing the layer-2 weights for the down-row units, setting each one's diagonal link and the bias that goes with it.

I'm now renumbering the layer-1 units: the pressure units sit at 0 to 7, the spill-to-self at 8, the upward recurrent units at 9 to 15, and the downward ones at 17 to 23. I'm also checking that each row's own-row units land at the right indices, and then I'll work out how each row's own contributions go into the layer-1 weights.

Now I'm setting the own-row contributions for each row, with the unit spacing of seven. For rows 2 through 7 that covers units 8+r, 15+r and 22+r, while row 1 only gets unit 9 and row 8 only gets unit 23, so unit 16 stays clean.

Still writing the own-row weights: each row gets its three recurrent units set in one strided statement per input channel, with the danger penalty applied to the far columns.

Still writing the own-row weights: the remaining statements set the channel 2 and channel 4 terms at the far columns for each row, and I'm checking that each slice lands on the right indices.

Still writing the spill weights: for each upper row feeding R_up_r, I'm setting the channel 1 target columns with one strided fill, then the diver and enemy columns with an explicit 14-value list that cancels the enemy pull.

Still working out the spill for each target row: the channel 1 cells in columns 0 to 5 get one strided fill, and I'm weighing whether to add the target-row units with a 2-D block so the differing distance values don't need separate statements.

Still working out the spill weights as 2-D blocks: I'm checking whether a single strided assignment can cover a whole source row's channel 1 cells across several target units at once, with the distance-dependent values laid out per unit.

Still settling the spill weights, I'm going to set each one as a scalar fill on a strided slice rather than as a 2-D block, since that's the safer form to write.

Still working out the spill weights for the target units, and I'm considering whether a single 2-D block could cover several target units at once, with the distance-decaying values laid out per unit.

Still working out the spill weights, I'm checking whether one per-pair set of three strided statements is simpler than a 2-D block, and estimating how the decaying logit values behave at a few typical distances.

Working out the spill weights, I'm computing the attenuated logit for each row distance: about 13.3 at distance 0, 12.0 at distance 1, and 10.4 at distance 2.

Checking the spill strengths at a few distances: a target at distance 5 comes out near 5.8 and at distance 6 near 4.8, both still well above the no-op value. A far target should still win out.

Now I'm checking whether firing in a hot row is worth considering, since the fire value there comes out to about −6 and that already looks like the wrong choice.

Checking whether a sub that somehow ends up outside column 8 breaks the policy: if no row is active, the output falls back to noop, and that case is rare enough to accept.

I'm weighing whether the spill toward distant rows is worth keeping, since the spill from row 1 would pull the policy up into row 2 when I only want it to go there from row 0. I'm also checking that the sub stays safe if it drifts sideways into water, where the fallback is mostly noop.

I'm now asking whether spill weights should skip row 1 entirely, since the policy never actually enters that row and a target there would only pull it up to row 2 where it just waits. I'm keeping row 1's own-row units in place for the cases where the sub is already in that row, and I'm treating the row-1 safety check the same way.

I'm narrowing the spill to rows 2 and up, since row 1 is never entered and a target there would only draw the sub to row 2 to wait. That leaves 15 pairs instead of the earlier count. I'm also weighing whether to allow upward moves into row 1 when oxygen is low, since then the risk of a sticky surface is tied to the diver count.

I'm now dropping the spill toward row 1, since the sub never actually enters it, and I'm checking whether allowing an upward move into row 1 when oxygen is low could be made safe. Then I'll work out what the policy does once it has descended into row 1.

Next I'm looking at what else could raise the score, like using the spare layer-1 unit 16 and layer-2 units 28 to 31 for a cooldown-aware fire rule. Firing with no target in the row is wasteful, so I'd keep noop preferred there, and diver pickups in my own row already push toward fire or hold. The own-row trail cells don't get any weight, so they shouldn't affect the policy.

I'm checking how an enemy in another column of my own row would interact with the policy, since a sub moving right toward my cell is already covered by the hot-cell check at columns 7 and 9. A target two cells away is treated as a hot cell and makes me leave the row, while one at column 5 counts as a target I should fire at right away.

I'm checking the enemy-in-my-row case: an enemy at column 5 moving right gets fired on when the cooldown is ready, and if the cooldown isn't ready I just wait, since it reaches the hot columns within a few frames and I leave before it arrives.

Enemy bullets in my row from the left at columns 0 to 3 also count as targets, so I'd fire and hold; they move one cell per frame, so the two-step margin before the hot cells at column 6 still works.

I'm checking the enemy-bullet case in my own row: a bullet moving left from cols 0 to 3 looks like a target and pushes me to fire, but my bullets can't hit it, so the row would keep reading as active and I'd be stuck there.

I'm estimating the score next: enemies pass through columns 0 to 5 for about 36 frames each, so I should catch most of them, and with a few divers picked up and a surfacing step, I'd guess somewhere in the 10 to 30 range. I'm also considering whether to surface a bit earlier than oxygen 60.

I'm reconsidering the surfacing risk and the difficulty ramp: enemies get faster after surfacing, so the hot-cell margin shrinks, but a one-to-two step buffer still seems to hold. I'm also checking the case of an enemy near column 8 moving into my row while I'm changing rows; the hot-cell rule at column 8 should cover that.

I'm checking the hot-cell cases for an enemy, a bullet, or a diver at the edges of my own row, and confirming each one blocks the move. Those cases are covered, so I'm turning to what happens when I move up into the next row.

Now I'm checking the enemy-spawn timing: a spawn at the far column only shows up after the step starts, and the spawn timer leaves enough time to move out of that row before it arrives. Vertical movement is safe on that front, since enemies only travel horizontally. I'm also noting that the random up and down moves in neutral rows are rare enough to accept.

I'm checking the enemy-spawn and sub-blocking cases, and I'm now tuning the layer-2 weights for the in-water down unit, since its bias is pulling the down signal slightly off. I'm trying a larger weight and bias pair so the water state separates more cleanly.

I'm recomputing the hidden-unit pre-activations for the water-state detector with the new weights of −6 and bias −39.5. In row 0 that gives about +8.3, which saturates to 1, while in water it gives about −3.7, which leaves the down signal near zero. So the separation holds.

I'm checking the gate margins for the X, Up, and Down units: active inputs land near −0.025 and inactive ones near −9.975, so the payloads can't flip a gate. I still need to recompute each unit's bias and weights precisely.

Now I'm setting the gate pre-activations for the X and Up units, with the active and inactive cases worked out, and fixing Up_1's bias at −4.5 so the surface payload comes out to about 2.79.

I'm now setting up the surface-payload gate for Up_2, adding the bias and weighting for the target-in-row-1 and surface cases so that each state lands on the right side of the threshold.

I'm now working through the per-row up and surface gates, where each unit's preactivation is a bias plus the sum of the row-state signals, and I'm checking which terms cancel out in the non-surface case. Then I'm setting the down-gate preactivations and the final logits for noop, up, and down.

Still writing the output weights and biases for the logit layer...

Still working out the output-layer constants for the down logit, checking the signs and the bias terms.

Still working out the output weights and biases for the logit layer...

I'm verifying the row-0 values for the policy: the x and a sums come out to −1 each, b to −1, and the SMU term at oxygen 200 reduces to roughly zero, so the start state checks out.

Now I'm checking the output values at the starting cell. With no enemy in the way, the noop, up, and fire logits come out to about 1.5, 0, and -6, and the down logit is about -16 unless the row below is clear, when it rises to roughly 7.6.

Now I'm setting the layer-1 weights for the per-row P units, each one watching the cell at column 8 in its row.

Setting the first-layer weights for the SMU unit: it reads two oxygen-related cells in row 9, adding a positive weight at one and a negative weight at the other, with a bias of −1.5 so it lands near ±0.9 in the two cases I've checked.

Still writing the first-layer weights for the row-0 unit and the up-left gate...

Still working out the first-layer weights for the spill terms on the upward-looking units.

Still working out the first-layer weights for the spill terms, with the pairs for the upward and downward units.

Still working out the first-layer spill weights for the downward units, pairing each source row with its target row.

Still working through the spill terms, checking the downward pairs and whether the row-1 safety case holds once I descend.

I'm weighing whether to add a refinement that detects a friendly bullet already in my row, so I can switch to another target instead of waiting, though that would need per-row gating I haven't set up yet.

I'm weighing whether a friendly bullet already in my row should make me hold off on firing, since the cooldown would block another shot anyway. If the bullet is heading to the target, staying put wastes time, so I'm considering letting neighboring rows pull me instead.

I'm defining a friendly-bullet detector that sums the bullet channel across columns 1 to 7 in every row, so it fires whenever a shot is still in flight. Bullets in the middle columns mean the cooldown hasn't reset yet, which tells me firing again would be wasted.

I'm trying to tell whether the row-1 ranking can use a single unit for the "bullet in flight" case, since layer 2 can't read the layer-2 X units directly. I'm weighing whether layer-1 row-own units can carry that gating instead.

I'm considering cutting the fire and noop logits when a friendly bullet is still in flight, so that staying put looks less attractive and the policy drifts toward neighboring rows or other targets. The catch is that the penalty would need to be large, and then the agent would wander randomly through neutral rows while a shot is in transit, which is probably acceptable since hot rows stay blocked.

I'm weighing whether X staying at +1 could become a problem, since a diver at column 9 moving right might keep the target signal active when it shouldn't.

I'm checking the diver cases now: a diver moving right at column 9 leaves a trail at column 8 that cancels out, so it stays neutral. If a diver lands on my cell while my divers are full, it shows up as a raw signal in my row and keeps the hold signal on, then it moves away. A diver passing through at column 7 leaves the trail at column 8 canceled as well.

Checking the down-penalty for surface mode: with SMU near −0.9 the gate is off, so the down logit stays suppressed as intended.

I'm checking the surface-mode down penalty against the new numbers, and the remaining edge cases look fine. Diver cases at rows just above me also resolve correctly: one directly ahead gets picked up, while one a column away makes me wait until it lines up.

I'm settling the first-step behavior at row 1: if nothing is in view, the hold signal keeps the agent in place, and I'm considering a small downward bias so an idle agent drifts away from the top edge.

I'm adding a small downward drift at row 1 and an upward one at row 8, so an idle agent doesn't get stuck at the edges, and checking that the hot-row blocking still holds with the extra bias.

I'm weighing whether to move up into row 1 instead of staying put in row 2, since row 3 is hot and holding still there keeps me exposed in my own row.

I'm checking the edge case where an enemy spawns in my row at column 9 with both neighboring rows clear, which should trigger an escape. I'm also watching for oscillation: after escaping to row y+1, the upward unit sees row y as hot and stays blocked, so that should hold. Then I'm considering enemies at columns 6 to 9 that are moving left, which will shift into the target zone once they pass column 8.

Checking the trail weights in my own row: an enemy at column 6 moving left just adds extra suppression, while a diver there cancels out to zero. A diver at column 8 moving right only partly cancels against the column-7 trail, leaving a small positive signal I still need to account for.

I'm writing out the per-cell contributions in the pre-scaled units, with the own-row trail cancellations for the wall and diver cases, and checking each value against the expected tanh outputs.

I'm checking each cell's pre-scaled contribution against the expected tanh outputs, and the enemy, bullet, and diver values all come out as intended. Next I'm working out the spill terms for the downward units, starting with the first distance step.

I'm working out the spill terms for the downward units, with the per-target and per-diver contributions in the neighboring rows. Next I'm checking how the layer-2 coefficient on R and the gating combine, and whether the own-row value for an enemy at column 4 or 5 still holds up.

I'm checking the own-row gating sums now: X comes out to about 0.954 at column 4 or 5, so the fire logit lands near 9.6. I'm also confirming that a target combined with an opposing bullet cancels to zero, which keeps the row neutral.

Now I'm checking the Up_2 unit's pre-activation, with the SMU weight and bias folded into a single expression for the non-surface case.

Checking the Up_y units for rows 3 to 7 now, where the pre-activation adds the row's own signal to the neighbor spill from below.

I'm working out the Up_8 drift, where a +0.5 bias on that unit's payload makes it lean toward moving up from row 8 when the row above is neutral. Next I'm setting up the Down_1 pre-activation, which combines its own row's signal with the spill from the row below and a fixed bias.

I'm now setting the Down_1 bias so that, with row 2 hot, the down move from row 1 comes out around −12.7 while the no-op stays near 1.5, and I'm checking that the row-1 hot case stays roughly balanced between moving and holding.

Still writing weight assignments for the layer-2 units...

Still setting the layer-2 weights for the downward units...

Still setting the layer-2 weights for the input-gating units...

Still setting the layer-2 weights for the downward and gating units, and checking the down-move signal at the top row.

Still checking the right-move gate for the water case, where the tanh values for the left and right units determine the logits.

Still setting the output logits for the movement and action units...

Now I'm working out the per-row logits for the water case, checking that the no-op comes out near 2 once the layer-2 sum is included.

Working out the down-move logit in open water now, where the row-1 unit is at zero and the bias pushes its pre-activation sharply negative.

I'm checking the enemy-above case in water at row 4: with a target in row 3, the up signal comes out strong enough that the agent moves up. At row 3 the fire unit dominates, so the agent shoots and the bullet goes out in its row.

Next I'm checking the case of a bullet already in flight: the target at column 2 still stays put, so I'll keep firing, and once it lands the target is gone and I move on.

I'm checking the bullets coming at me from the enemy sub: if its shot is still a few columns out and the sub is alive, the net signal in my row is hot once the bullet gets close, so I'd need to move, and while it's still far away I can just hold and keep firing.

I'm sanity-checking the Up_2 and Down_1 cases and confirming that row 1 stays balanced, then I'm going over the escape rule for row 8, which can only move up.

Now I'm checking the surface-mode case at low oxygen, where the upward units get a boost that makes neutral rows above look attractive, so I'll climb row by row.

I'm checking whether the surface-mode fire priority needs lowering: with oxygen under 60 and several rows holding targets, waiting out bullet flight in each could eat the air budget. I'm estimating about 40 steps against the 60 available, but the up logit doesn't yet beat noop once fire is suppressed, so the weights would need adjusting.

I'm checking whether raising the surface threshold would help, but g3 (oxygen under 80) would trigger surface mode whenever six divers are collected, which would leave the agent stuck until it runs out of air. The original g2 threshold of 60 avoids that trap, so I'll keep it.

I'm settling on the surface-mode fire problem: scaling the own-row signal down by a constant c in the surface case, and checking c = 1.5, which zeroes the fire pull so the upward move wins.

I'm checking whether a neutral row in surface mode would look hot once the fire penalty is applied, and it does, so the escape logic would fire. Down would also get the +14 bonus, but the −5 surface penalty keeps it below up when the row below is empty, so the agent should move up about 90% of the time.

I'm checking whether surface mode could get stuck, since fire at a target row could keep winning over moving up, and I'm weighing whether that matters given the 60-step oxygen budget.

I'm checking the case of a diver at column 9 moving right in our row, and whether it could keep us stuck in place.

I'm checking the trail cancellation at column 8 in our own row, and confirming that the enemy trail on the left adds the expected extra weight. I'm also confirming that our own back-trail at column 9 carries no weight.

I'm checking that row y's own presence stays invisible to the policy, since the own-row signal doesn't include the channel that marks me. The hot penalty is strong enough that a single hot row drives the target signal down to about −1. Idle moves into neutral rows at a few percent each direction should be harmless, and idle firing just sends bullets down my own row.

I'm checking the edge case where a neighbor row is neutral but the row beyond it is hot, which would pull me into that neighbor and then leave me waiting there, and that seems acceptable. I'm also confirming that a hot neighbor always dominates even if it has a target in it. At row 2 with row 1 blocked, the up signal drops to about zero, so the down move into the neutral row 3 wins.

I'm checking the spill from rows above and below: targets in the rows beyond mine feed into the neighboring rows, but my own row stays out of that calculation. I'm also confirming that the row-above and row-below signals each only see the rows past them.

Writing the layer-1 weights for the own-row units now...

Still writing the layer-1 weights for the sub-row units...

Still writing the layer-1 weights for the own-row units...

Still writing the layer-1 weights for the own-row units...

Still writing the layer-1 weights for the own-row units...

Still writing the layer-1 weights for the remaining row-mapped units...

Still working through the remaining weight slices for the own-row units, and wondering whether writing each unit separately would be safer than relying on stepped slices.

Still writing the spill weights for the upward units...

Still writing the spill weights for the downward units...

Still writing the spill weights for the upward units...

Still working out how the spill weights from the rows above and below feed into the upward and downward units.

Still working through the spill weights and the stepped slices, and double-checking that the enemy-trail cancellation at cols 7..9 is worth keeping.

Still checking the enemy-trail weights and the early-game path; I'm moving on to the mid-game hunting behavior next.

Still working through the spill weights, and checking how a bullet at column 3 moving right would look to the policy when I enter its row.

I'm checking the bullet at column 5 in a row I'm about to enter: with the shooter still active and the bullet closing in, the hot signal keeps me out until it passes. Even if I'm stuck holding for a step, the bullet reaches column 8 before I'd be caught, so the two-level timing still looks safe.

Checking the sticky-action case: two repeated holds in a row is about 1% likely, and staying put while a bullet is coming has a tiny softmax probability, so the timing still looks safe. Next I'm looking at a fish moving right into my row: once it reaches column 6 the row turns hot and I leave without taking the shot. Then the case of two enemies in one row, where I'd kill the nearer one first and then the other.

I'm checking the diver case at column 5 moving left, where the trail at column 6 gives a strong target signal, so I'd hold for about twelve frames until it reaches column 3. That's a wasted stretch every time a diver passes like this, so I'm weighing whether the weights need adjusting.

I'm working out the diver weights for the case where a diver is at column 4 or 5 moving left, since the trail would currently make me want to wait for a diver that's just going away. Adding a negative trail weight at those columns would make me avoid the row instead, but I'm checking how that shifts the signal for the diver at column 4.

I'm weighing a smaller trail weight at columns 5 and 6, since a full negative weight would make enemies at columns 4 and 5 read as hot and push me out of the row. A milder value of −0.75 would leave a fish at column 4 neutral and a diver at column 4 still mildly attractive.

I'm weighing whether a single linear weight can separate a moving-left enemy from a moving-left diver at the same columns, and it looks impossible: the enemy needs a positive total, the diver a non-positive one, and the bullet a negative one, so I'll have to prioritize which case to get right.

I'm checking whether divers moving left through columns 4 and 5 will pull me toward their rows, since the own-row attraction from neighboring rows would catch them. Most divers pass through there, so this could cost me some frames, but it's an acceptable tradeoff.

I'm rechecking the enemy spawn cases at the edges: an enemy entering at column 0 moving right counts as a target right away, while one spawning at column 9 moving left in my row only shows up once I'm already there. I also want to confirm that a bullet trailing behind me at column 9 reads as hot rather than as a target, and it does.

I'm checking the idle case: with no targets or threats in view, I hold position, and near the top or bottom rows the drift keeps me from getting stuck. The hot penalty on the up neighbor is strong enough that moving toward it is very unlikely, and left or right moves in the water are similarly suppressed.

I'm checking the up-action payload and its active pre-activation for the next neighbor, which comes out near neutral, and then confirming the up-blocked units are firmly negative. I'm also turning to whether surface mode can happen at row 0 after a surfacing with a positive dive count.

I'm double-checking the oxygen-threshold logic: below 60 with no divers collected, the policy keeps hunting, and once a diver is picked up the surface mode turns on and sends it upward. I'm also verifying the Down_y index mapping in the second layer.

Next I'm checking the index layout for the second hidden layer: the up units sit at 8 through 15 with their output weight of 14 and bias offset of 7, the X units at 0 through 7, and the special units at 23 through 27. I'm also noting that layer-1 unit 8 and layer-2 unit 23 share an index number but live in different layers, so that's fine.

Still writing the second-layer weight matrix...

Still writing the second-layer weights for the Up and Down units...

Still writing the second-layer weights...

Still writing the third-layer weights...

Still writing the third-layer weights...

Checking the rightward move at column 9: the back cell at column 8 and the front cell at column 9 combine to a strongly negative value, so right comes out heavily suppressed. The left move at column 9 still wins clearly.

Checking the down-action logic at column 9 next, and confirming the trailing-bullet cases on both sides of the policy.

I'm checking the sticky-action edge at the start of an episode and the spawn position for the sub, then confirming the row-1 contribution only feeds unit 9.

Still writing the spill weights for the Up units...

Still writing the first-layer spill weights for the Up units...

Still writing the first-layer spill weights for the Up units...

Still writing the first-layer spill weights for the Up units...

Still writing the first-layer spill weights for the Up units...

Still writing the first-layer spill weights for the Up units...

Still writing the downward spill weights...

Still writing the first-layer weights for the Down units...

Still writing the first-layer own-row weights for row 2...

Still writing the own-row weights for row 2...

Still writing the third-layer weights for row 3...

Still writing the third-layer spill weights for the lower rows...

Still writing the weight index ranges for the bottom rows...

Still writing the spill weights for the Down units...

I'm checking the bullet case at column 5 in a neighboring row: the bullet's negative weight is mostly offset by the spill from the rows below, so the net signal comes out slightly positive. That would pull me toward moving up, but the bullet moves right into column 6 on the next step, so I'd have time to step away again.

I'm checking the sticky-action fallback: if a move brings me into a row with a bullet, repeating that move carries me further up and out of the bullet's path. The remaining concern is whether the tanh activation on a slightly positive pre-activation is safe enough to commit to.



---

I'm picking up where the last reply got cut off, but the earlier content isn't visible to me, so I'll rebuild a compact design brief and the edits block. I'm working out a simple submarine policy: the sub starts at the surface row, and staying there doesn't end the episode, but moving down and then returning to the surface without a diver does.

I'm weighing a simple dive-and-shoot policy against staying at the surface: staying at row 0 is safe but earns nothing, since enemies only spawn in rows 1 to 8 and oxygen runs out after about 200 steps without a surfacing. Hand-designed policies should beat the few points typical of trained MLP agents, so I'm aiming for a reasonable middle ground.

I'm weighing a fixed-column idea next: since enemies spawn at the left and right edges of rows 1 to 8, keeping the sub at a predictable spot would make the features easier to design, though it means dodging costs movement I'd rather avoid.

I'm settling on a fixed row and column for the sub so the features stay predictable, then firing toward whichever enemy comes close. Turning costs a move, though, and the enemy timing depends on the move speed, so I still need to work out when an approaching enemy will actually reach the sub.

I'm estimating how many enemies will show up in my row over the roughly 200 steps before oxygen runs out. Enemies spawn about every 21 steps, spread across eight rows, so each row should see only a handful.

I'm estimating roughly 1.25 enemies per 200 steps in my own row, so shooting only horizontally would earn very little. To score more, I'd need to move vertically to line up with enemies, and surfacing with divers could extend the episode. I'm also recalling the MinAtar Seaquest baselines, which put trained DQN at roughly ten or so points, though I'm not certain of the exact figures.

I'm thinking about a scheme that computes features relative to the sub's position, with one unit per row to detect an enemy in the same row. Working out the left-or-right direction is harder, so I'm considering a vertical alignment strategy instead: move to a row where an enemy is, face it, and fire.

I'm thinking about keeping the sub in a fixed column and never moving left or right, so its bullets only fly one way. That leaves enemies from the other side as a threat I'd have to dodge vertically, and I'm checking whether a fixed column at the edge helps.

I'm checking the spawn edge case: an enemy appearing at my own cell on the right edge would only hit me if I stayed put, since I can step up or down before the enemy update checks for collision.

I'm checking the spawn risk for an enemy appearing on my cell at the right edge: about 1/16 per spawn, so roughly half the chance of dying over a full run. Right-moving enemies start on the left and I face left, so I can shoot those, but left-moving enemies spawn on me when I sit at column 9.

I'm checking whether a sub parked at the right edge, x=9, could still be safe from left-moving enemies that spawn right on top of it. Enemy bullets fly through our bullets, and an enemy sub in our row firing toward us is a real threat. Moving to x=8 facing left would leave a left-moving enemy spawning at x=9 behind us, where we can't shoot it.

I'm checking whether a fixed column at the left edge would work: pressing left from x=5 five times lands me at x=0 facing left, and then a right press turns me around.

I'm checking the position at x=1 facing right, which leaves x=0 behind me, since facing the wall at x=0 would make my shots useless. Enemies moving right would spawn at x=0 and reach me, and I can't shoot them from that orientation, so I'd have to dodge them vertically.

I'm weighing a fixed column in the middle, around x=5, where enemies from behind would take a good while to arrive and I'd have time to dodge. Turning toward an enemy costs a move, though, and telling whether it's to my left or right means comparing my column to the enemy's, which is harder to build into the features with a fixed column.

I'm noticing that keeping the sub on one fixed column doesn't actually hold: turning around means stepping one cell over, so the sub just alternates between columns 5 and 6. That means its position reduces to a row index plus which cell is the front, which is a much smaller state to build features around.

I'm checking whether the sub can stay in columns 5 and 6 by turning back and forth, since that keeps its position to just a row index and an orientation. Sticky actions could push it out to column 4 or 7, so the policy needs a correction that pulls it back toward the middle.

I'm settling on keeping the sub in columns 4 to 6 and letting it drift back toward the middle if sticky actions push it out. Instead of hard-coding columns, I'm thinking about per-row features that tell the policy whether an enemy sits left or right of the sub in its row, and using a small recentering term to pull it back.

I'm working through how to split the 16 per-row side units, and whether restricting the sub to a few rows would save capacity, since enemies still arrive in all eight rows. I'm also turning to what actually earns points: each kill is worth one, and surfacing with divers could extend the episode.

I'm weighing how the score actually accumulates here: each kill is worth one point, and surfacing with divers could add up to ten more by refilling oxygen and letting the episode run longer. Without surfacing, the episode is capped near 200 steps with only about nine enemies spawning, so a realistic aim policy might land somewhere around three to six kills.

I'm sketching a row-hunter strategy: move vertically toward rows that hold an enemy or diver, fire when an enemy lines up in the sub's row, and turn toward targets when needed.

I'm settling on a potential-style comparison for the vertical choice: the difference between the up and down logits should track the gap between the sub's row and the target rows, since that's linear in the inputs.

I'm sketching the first layer as per-row sub-position indicators, one unit per row that fires when channel 0 has the sub's cells in that row, so the second layer can combine them with enemy features.

I'm checking whether a single layer-2 unit can detect "enemy in my row," since the sub's row indicator and the enemy row indicator have to be combined and a plain sum over rows doesn't capture that conjunction.

I'm sorting out how to detect enemies in the sub's row: one unit per row would catch an enemy to the left, but that's 16 units in the first layer, and vertical targeting will need comparisons on top of that.

I'm looking for a vertical rule that doesn't need a conjunction per row, and I'm testing whether a single linear unit over the enemy cells' row positions can give the direction I need.

I'm weighing a fixed-row strategy that shoots in both directions and only dodges when needed, but that would catch only about 1.25 enemies over 200 steps, which is poor. I'm also checking what realistic scores look like on MinAtar Seaquest, where small agents tend to land somewhere around 5 to 15.

I'm laying out the per-row units next: one pair for each row from 1 to 8, each firing when an enemy sits to the left or right of the sub in that same row, which uses 16 of the first-layer units and leaves the rest for vertical targeting and danger.

I'm looking at a thermometer-style encoding for the sub's row, with one unit per threshold so the second layer can ask whether an enemy sits above or below it, though that still needs a conjunction per row.

I'm checking whether the enemy-above and enemy-below detectors can each be a single conjunction unit per row, which gives 16 units for vertical targeting. Adding the horizontal left and right units brings the total to 32, which leaves no room for danger, oxygen, or surfacing, so I'm weighing how to combine them.

I'm weighing whether a graded version of the above-and-below detectors could handle enemies on both sides, with closer rows counting more than distant ones.

I'm weighing how the sub avoids danger, since an enemy or bullet reaching its cell would end the run. Firing down the row should kill most approaching enemies first, because a bullet moves one cell per step while an enemy only moves about once every six steps, so the bullet should land on it before it reaches the sub. I'm also checking how the bullet's starting position relative to the enemy affects whether that hit lands.

I'm checking how enemy bullets interact with the sub's own shots: our bullet kills an enemy sub in its path, but any bullets already in flight from that enemy keep moving toward us. Enemy subs fire on a fixed interval, so the first shot from a freshly spawned sub comes after a predictable delay.

I'm checking how an enemy bullet already in flight behaves when its sub is killed, since it keeps moving toward us. A sub that spawns from the side needs about eleven steps before its first shot, so it usually dies before firing. The real danger is a bullet already in our row heading toward us, which would mean moving vertically out of that row.

I'm checking the vertical danger case: an enemy at the cell I'd move into, or one about to slide into my column at 5 or 6. Only my front cell can be hit, so I'd treat danger in columns 4 to 7 of the target row as blocking the move, and I'd need a conjunction unit per row to detect it.

I'm trying a layer-1 setup that computes per-row features without conjunctions, such as sub-in-row, enemy-anywhere-in-row, and danger in the center columns, and then letting layer 2 handle the conjunctions. That would avoid spending a unit on every row-pair combination within the budget.

I'm trying to fold the horizontal left-right detection into the vertical logic, so each row-specific unit carries a signed sum over enemy columns and only fires when the sub is in that row. The budget is still tight, since that leaves per-row costs for vertical attraction and vertical danger on top of it.

I'm making each row's horizontal unit saturate at -1 when the sub isn't in that row, and only carry a signed left-versus-right enemy value when it is. Summing those units across rows then gives the sub's horizontal direction, shifted by a constant from the rows it isn't in.

I'm settling on one unit per row for the horizontal direction: it carries a signed left-versus-right enemy value when the sub is in that row and sits at -1 otherwise, so summing across rows recovers the direction. The same gating idea could extend to vertical attraction, with one unit per sub row weighting enemies in other rows by their signed distance.

Checking the gating idea a bit further: each row's unit is pushed to -1 whenever the sub is elsewhere, so only the row it's in contributes a real signal. For vertical attraction I'd weight other rows by their signed distance, with the nearest rows dominating. The open question is whether chasing enemies alone is enough, or whether divers in the reward channel should pull the sub toward them too.

I'm checking whether divers can be used the same way as enemies for vertical attraction, since a diver in my row drifts across the columns and reaches my front cell at 5 or 6, so picking one up only needs me to be in its row. Friendly bullets don't touch divers, so I can wait in that row while firing at enemies as needed.

I'm now considering whether horizontal turning should follow the reward channel, which treats divers as attractors too, or the danger channel, which only tracks enemies and bullets. Turning toward enemy bullets wouldn't help, since facing doesn't stop them, but a bullet in my row heading at me is a case for dodging vertically instead. I'll lean toward the reward channel for turning and accept that divers pull a little as well.

I'm sketching the firing side now: I want to shoot whenever an enemy sits in my row ahead of me, and to know which way I'm facing I'd use a unit built from the channel-4 trail cells at my two columns, though enemy trails in other rows could throw that off.

I'm weighing whether orientation can come from the trail cells at columns 5 and 6, but enemy trails in other rows would add noise. Pressing left just to turn would also drift the sub out of its columns, so I'm looking for a way to know facing without that cost.

I'm stuck on the orientation problem: if the sub keeps pressing left whenever an enemy is to the left, it can drift into the enemy. I'd need to know which way it's facing, and the trail cells that would tell me are noisy because enemy trails in other rows show up in those same columns. Gating the per-row turn signal on facing might solve it, but that still needs a reliable facing read.

I'm checking whether the orientation can be read from the back-cell trail, meaning cells that are both marked as my body and carry a trail, since a front-cell trail would normally mean an enemy just passed through. Enemy trails in other rows at columns 5 and 6 could still add noise, so I'd need a per-row gated reading to keep that from skewing the facing estimate.

I'm looking at a per-row orientation unit that uses the sub's back-cell trail to tell which way it's facing, with a bias that pushes it toward -1 in rows the sub isn't in. The open question is whether I can fold that into the per-row horizontal units rather than adding eight more.

I'm weighing a per-row "wanted action" signal instead of folding orientation into the horizontal units: if the sub faces right and an enemy sits to the left, it should turn left, and if it already faces the enemy it should just fire.

I'm working out how to encode the three cases for turning and firing in a single gated signal: turn left when the desired and current facing differ one way, turn right when they differ the other way, and fire when they match. The catch is that a gated-off row saturates at -1, which collides with the turn-right value, so I'll need to separate those two outcomes.

I'm working out how to make the per-row unit give a three-way signal: turn left, turn right, or fire. Gating the rows that don't contain the sub at -1 keeps the sum recoverable, since the sub's own row is the only one that varies.

I'm comparing the desired facing with the current facing, so a per-row difference of -1 or +1 means the sub is off by one step, and that difference would tell it to turn. I'm worried a tanh of the difference would saturate and blur the one-versus-two cases, so the scaling needs care.

I'm trying to simplify the design by taking the enemy count from channel 2 alone, since mixing in divers and bullets makes the left-minus-right difference harder to read. Since the spawn rules keep enemies in a row moving the same way, I'd expect one or a few enemies per row, so a clipped count should be enough.

I'm checking whether the per-row units can just count enemies to the left and right in each row, since the sub's column is fixed, so "enemy left of me" only needs the row-specific gate. I'll accept that a diver on one side and an enemy on the other might cancel out, since that case looks rare.

I'm settling on nine per-row units, rows 0 through 8, and dropping the idea of skipping row 0. The catch is that when the sub sits in row 0 the saturated values sum to something that looks like a turn-right signal, so I'll let the dive unit take priority there and treat row 0 as a special case.

Now I'm working out how the vertical movement should be decided: the sub should prefer diving at the surface, and in other rows it should weigh where the divers and enemies are.

I'm weighing a dodge rule for enemy bullets in my own row: a bullet heading toward the sub from the side is a reason to move vertically, not to fire.

I'm considering a dodge for enemy bullets in my own row: if one is coming at the sub from the side, I should move vertically rather than fire. The catch is that moving up from row 1 reaches the surface, which is only safe if I have divers, so I need to check the diver count in the bottom row's gauge before choosing up.

I'm checking how the oxygen and diver counts share row 9: the oxygen bar fills from the left and the diver bar from the right, so they overlap. Reading the diver count only works once the oxygen bar is short enough to leave the rightmost column free, which means it matters mainly when oxygen is low, the same case where surfacing is needed.

I'm now checking the oxygen and diver bars in row 9: oxygen fills cols 0 to 2 at low levels, and the diver count only shows up in col 8 once oxygen drops below about 180. So the bottom-row readings give me a clean surface rule.

I'm sorting out how surfacing should work: go up only when oxygen is low and a diver is on board, and keep the sub from rising to the surface from row 1 otherwise.

I'm weighing how fast the policy could score: enemies arrive roughly every 21 steps and each takes about 60 steps to cross, so chasing the nearest reward row and firing should yield many kills, and I'm estimating how high that rate could go.

I'm estimating how fast the policy could score: the spawn rate is about one enemy every 21 steps, so catching that rate would give roughly nine kills per 200 steps. Surfacing with divers should extend the episode, and the difficulty ramp will make things faster over time. The main risks are collisions when moving vertically and enemy bullets, so the danger blocking has to be careful.

I'm checking how enemy bullets behave in my own row: when I turn left, my front cell moves into col 5, and an enemy or bullet arriving from the left could meet me there, so the timing matters. Enemy submarines fire bullets toward me along their row, and killing the shooter doesn't remove bullets already in flight, so I still need a dodge rule for those.

I'm checking whether an enemy bullet in my row close on either side should trigger a dodge, since I can't tell which way it's travelling and dodging costs little. I also need to watch the timing of a bullet two cells away, because it could reach my position on the step after the next.

I'm adding a danger check for enemy bullets in the sub's own row, looking at the cells on either side of the sub, since a bullet close to the front cell can hit on the next step. I also want to flag nearby enemies in that row as a separate danger.

I'm checking vertical moves now: before going into a new row, I'll block the move if any enemy, diver, or bullet sits in the front columns there, or in the columns just beside them, since those could move into my cell on the next step.

I'm now weighing how to fit the dodge into the per-row vertical units, since a signed value can't express "move vertically, either way." I'm considering a separate magnitude-style signal for that, but the unit budget is tight with 16 units already taken.

I'm wiring up the gating so each row's vertical unit is switched off when that row has no sub in it: a large negative offset on the row's presence signal, which lets the content term dominate when the sub is there. That makes the per-row values sum cleanly to the sub's row position, and I'm checking that the row-0 case still comes out right.

I'm reconciling the gating with the row-0 case: when the sub is present, the content term drives each row's vertical unit, and when it's absent a large negative offset pushes that unit toward -1. That leaves the sum at -8 for row 0, which reads as "down," the answer I want there. I'm now deciding whether a second hidden layer is needed at all, or whether these units can be kept near-linear enough to route the logits directly.

I'm setting the action logits next: fire gets a baseline, and the up and down choices scale with the vertical unit, so a clear vertical preference beats firing and a weak one doesn't.

I'm settling the priority order for the action logits: surfacing when oxygen is low and a diver is aboard comes first, then dodging, then turning toward an enemy in my row, then firing at it, then vertical moves, with firing as the default. Because a wrong move near the surface or in a bullet's path ends the episode, I want the logit gaps large enough that sampling is almost deterministic. When an enemy sits in my own row, I'm holding off on vertical movement so I don't drift out of the line of fire.

I'm trying a variant where the own-row enemy signal lives in its own unit instead of the signed vertical value, so that "stay and shoot" can outrank vertical moves. That means raising the fire logit only when an enemy is actually in my row, and keeping the baseline fire logit lower than the vertical ones otherwise.

I'm checking whether the row's presence signal can carry the enemy-in-row information too, by looking at the sign and value of D minus F for each row.

I'm defining a fire detector for the sub's own row: tanh units on the summed row-presence signal, with thresholds at ±0.23, so that a near-zero value means an enemy is straight ahead.

I'm weighing whether the enemy-count term can saturate, since two enemies in the row shouldn't read the same as one. A single linear unit mixing the count and the facing flag can't do that cleanly, so I'm looking at scaling the facing term more strongly to keep the fire decision stable.

I'm checking whether one enemy on the left should count as "turn left" or "fire," which depends on how the orientation term combines with the count. Two objects on the same side should saturate rather than add, so I'm looking at computing the counts per row before they get combined with facing.

I'm now checking whether orientation can be read globally from columns 5 and 6 across all rows, using the ch0 and ch4 channels together. The catch is that trails and other objects in those columns could contaminate the signal, so a single linear unit may not separate them cleanly.

I'm checking whether the orientation can be told apart from trails and other objects in columns 5 and 6, since a single linear unit summing across rows won't separate them cleanly. I'm now looking at whether a threshold on the combined channels could work, though that would need a per-cell nonlinearity, which per-row units can supply.

I'm weighing whether a single threshold can separate the fire case from the turn cases when the count of enemies can reach two, since a linear unit over D and F only works cleanly if D stays bounded.

I'm checking whether a global orientation cue can come from the sub's back cell in columns 5 and 6, using the ch4 channel at those positions summed across rows. The worry is that trails from other objects in those columns would contaminate the signal, and with a handful of objects on screen that happens often enough to matter.

I'm now weighing whether the sub can avoid depending on its orientation at all, since orientation is hard to read cleanly from the board. One option is a fixed alternating horizontal pattern, though it's unclear whether that can respond to threats in the right direction. I'm also considering whether restricting the sub to fewer rows would free up enough units to keep the per-row signals separate.

I'm considering whether orientation can be read per row, combined with the enemy-in-row signal, rather than from a global cue that trails in columns 5 and 6 could contaminate.

I'm sketching the layer-1 units for the rest of the rows: an orientation unit per row that reads the back-cell channels, plus a horizontal-desire unit and a vertical unit per row, with the rows summed in layer 2 into scalar signals whose conjunctions give the turn and fire decisions.

I'm now seeing whether the sub's facing can be folded into the horizontal signal instead of kept separate: if the left-minus-right enemy count is squashed into a signed value, the fire and turn cases might split apart in layer 2 with a simple combined threshold.

I'm checking the decision thresholds for each direction: fire-right, turn-left, and turn-right all split cleanly with a 1.5 margin. When the sub is on the bottom row, every row reads as an enemy on the right, so I need a row-0 indicator to suppress that false fire signal and push the sub down instead.

I'm making the per-row horizontal signal conditional on the row actually holding a sub, using a gate of -G when it doesn't, so absent rows stay pinned near -1. I'm setting the gate size so it dominates the count term, then checking that the sum over rows still decodes cleanly into the sub's horizontal value.

I'm working out whether counting enemies with both channels, and divers with only one, makes a tanh on the left-minus-right difference usable. If an enemy is on one side it should clearly register; a lone diver only gets a weaker signal, which I still need to check against the presence threshold.

I'm considering whether to exclude divers and bullets from the turn-and-fire signal, but a single symmetric unit can't threshold both sides, so I'll accept that they trigger it too. I need dodging to override that, since a bullet in my row shouldn't keep me sitting still.

I'm narrowing the enemy count to the fish channel alone so divers still register but bullets drop out, and I'm skipping the trail channel for direction. I'm also questioning whether columns 5 and 6 should be excluded from the count, since a diver sitting on the back cell only gets picked up once it moves forward.

I'm checking how the sub's horizontal position relates to its facing when it drifts, since pressing left or right can shift which columns hold the front and back cells. If the sub settles into cells 4 and 5 instead of 5 and 6, the column split still works, so drift is a minor concern rather than a blocker.

I'm checking the front/back split again, since facing left puts the back cell to the right of the front and my ch4 difference had the sign flipped. I'm now weighing whether a column-weighted sum over ch0 and ch4 can recover facing direction regardless of whether the sub drifts to cells 4 and 5 or 5 and 6.

I'm thinking about whether the column-weighted sum can recover facing direction, since a back cell sitting to the right of the front gives a positive value. The catch is that enemy trails in our row would add large weighted contributions wherever they sit, so I'm checking how much that contamination could matter.

I'm narrowing the trail and orientation check to columns 4 through 7, where the sub's cells usually sit, and weighting each column by its distance from the middle so a single expression can tell facing left from facing right. The open question is whether the trail contamination stays small enough inside that window to leave the signal usable.

I'm checking the facing sign on the two placements: with the sub at cells 5 and 6, the back-cell weights give +1 when facing left and -1 when facing right, and at cells 4 and 5 the facing-left case also comes out positive, so the weighting holds across both spots.

I'm checking whether the sub can stay in cells 5 and 6 instead of drifting, since a sticky repeated left turn would shift it to cells 4 and 5. The facing sign still holds for that placement, and the trail contamination only matters when an enemy sits close to the sub.

I'm accepting that the sub may drift, and I'm wondering whether a mild recentering term in layer 1 could push it back toward the middle when no enemy is in its row. The open question is whether the left and right column splits can stay fixed at 0..4 and 7..9 without mismatching after drift, or whether I need something that tracks the sub's position more directly.

I'm now worrying that the drift from repeated turns will push the sub off the column split I assumed, so I'll check whether a larger logit gap keeps random actions from creeping in too.

I'm working out the vertical pull for the sub: in each row other than its own, a reward pulls it up or down depending on whether that row is above or below. I'm also considering whether enemies and divers should pull with the same weight, and I'm making sure the sub doesn't climb into row 0 by accident, since from row 1 any upward push would take it to the surface.

I'm now weighing how the vertical pull should scale against the horizontal signal, so a reward cell at a farther row contributes less than a nearby one. I'm writing that as a distance-weighted sum over reward counts per row, passed through tanh so the vertical output stays bounded.

I'm adding a blocking term to the vertical signal, so danger in the row above the sub's columns 4 through 7 pushes it away from going up, and danger in the row below pushes it away from going down. If both directions are blocked the attraction term could still pull it into danger, but I'm treating that as rare enough to accept.

I'm checking how an enemy fish one column left of the sub's front would interact with a move up, and whether the sub could be caught in the fish's path.

I'm checking how blocking should treat the sub's own row, since an enemy bullet can only be told apart from an enemy by the second channel. I'm weighting the danger zone around columns 3 through 8 so that the central columns count fully and the outer edges count half.

I'm weighing whether the dodge signal should count each bullet in the sub's own row, and I'm noticing that a diver there would push the sum negative and cancel it out. I'm also checking that pushing down stays sensible unless the sub is already on the bottom row, and that the dodge only matters when no enemy is engaging horizontally.

I'm settling on a dodge priority that outranks the fire-and-stay response, since an enemy bullet in our own row would otherwise hit us while we shoot. I'm thinking about keeping the attraction term small so the dodge term can dominate when it matters, and giving the dodge a large weight so it wins in those moments.

I'm working out an urgent-down signal from the vertical pull: when the summed value sits near the top of its range, the sub should move down, and I'm checking whether a steep tanh with an offset near 0.98 can separate that from the ordinary attraction. Noise from absent rows barely shifts the result, so a threshold there looks workable.

I'm weighing whether to fold the front-cell signal into the per-row dodge units, since a single global count would pick up trails from every row and mostly add noise. Keeping the back-cell test local to the sub's own row looks cleaner, though it means I have to find room in the budget for one more set of per-row units.

I'm folding the front-cell orientation signal into the per-row design, but a global count would collect trails from every row and mostly add noise, so I'm weighing whether a reward-count difference in the sub's own row, combined with an orientation term, can keep the signal clean enough to use.

I'm comparing the sign combinations of the left and right reward counts with the facing term, and I'm checking where each case lands on the line. Fire happens at the two extremes, turns sit in between, and no action stays near zero when both counts are empty.

I'm checking whether two reward cells on the left would push the signal past the fire threshold even when the sub is facing the wrong way, which would make it fire when it should turn. The enemy approaching from behind is the risk I'm weighing against that.

I'm checking the tanh values for the turn and fire cases: with a=1 and b=0.3, the fire cases come out near 0.86, turns near 0.6, and no-action near 0.29, so layer-2 thresholds around 0.45 and 0.75 should separate them. I'm also noting that two reward cells on one side would push the sum into the fire range even when the sub faces the wrong way.

I'm weighing whether the vertical pull and the horizontal facing signal can share the same units, since that would free up room for a separate dodge term. Using a weight where the facing term is smaller than the reward term keeps the decoded cases apart, so I'm settling on that and giving the own-row dodge its own per-row units.

I'm settling the per-row layout: the vertical units are signed, but the own-row dodge can use a separate gated unit that sits at about -1 with no bullet and climbs toward +1 when one is present. Summing those gives a clean baseline, so I can read the dodge signal without worrying about row offsets.

I'm weighing whether the dodge direction should come from the vertical signal itself, so an upward reading sends the sub up and a downward one sends it down. The catch is that when that signal sits near zero, the direction is ambiguous, so I'm considering a slight downward bias from own-row bullets folded into the vertical content.

I'm settling the dodge direction: the signed vertical signal picks up or down, and I'm counting the same dodge urgency in both logits so that the difference decides. When the vertical signal sits near zero, the choice is roughly even, which is fine since both moves escape the bullet. The one exception is row 1, where moving up would surface the sub without a diver, so I'm planning a separate indicator that applies a strong penalty to the up logit there.

I'm settling the row-1 case: moving up from there would surface the sub without a diver, so the up logit needs a strong penalty, and only a surfacing condition with divers present should override it. I'm checking that the diver signal holds, since it only reads reliably once oxygen drops below the low threshold. I also want the sub to head down from the surface row, and I'm wondering whether the surface flag should persist after it has returned up.

I'm checking the oxygen threshold: column 2 turns off once oxygen drops below 60, so the surface flag still works there. I'm also noting that moving left or right and firing stay safe while surfacing, and that an up move from row 1 without a diver is the case that needs the penalty.

I'm making surfacing override the other signals, but moving up into a blocked row would still be a collision, so I'm weighing whether a large surfacing term can coexist with a strong negative reading from the blocking signal.

I'm checking whether a strong surfacing term can still lose to a blocking signal above, since moving up into a blocked row would be a collision. I'm considering giving blocking a large negative weight in the vertical signal so it saturates near -1, while keeping attraction small enough that it can't mask the block.

I'm checking the blocking threshold numerically and it's too sensitive: the blocked and unblocked values sit close together, so I'll limit the attraction term to about ±1 so the gap widens and the block signal stays reliable.

I'm weighing a linear version instead of thresholds: up gets 6 times the vertical signal and down gets the negative of it, so a blocked row above pushes the sub hard downward. Attraction would then need to stay moderate so it doesn't overpower that block.

I'm thinking about splitting the vertical signal into separate up and down units per row, since a single signed value would push the sub down whenever a blocking cell sits above it. That would take 16 units, but it keeps the "stay put" option open when the target row is blocked.

I'm trading off a separate dodge unit against blocking: with 32 units needed and only 24 available, I'm weighing which signal matters more. Blocking matters because the sub moves vertically a lot and must not run into enemies, while dodging matters because enemy bullets travel along the row. I'm leaning toward keeping blocking as the priority and handling dodge with the remaining capacity.

I'm weighing how the dodge and blocking signals fit together in one vertical value, since blocking matters most for the vertical moves and dodging matters for bullets that are already in flight along the row. The next question is how engagement priority interacts with those terms.

I'm weighing a steeper gain on the vertical signal so that dodging wins over engaging, but at these gains the fire and vertical logits end up too close to separate cleanly. I'm considering a layer-2 threshold unit to make the choice crisper.

I'm defining two layer-2 threshold units on the signed vertical value, one for strong up and strong down, plus weaker versions for mild preferences. The offset of 7 makes row 0 come out near -1, which conveniently already favors moving down from the bottom.

I'm now laying out the logits as a priority ordering: strong vertical dodging at 12 sits above turning at 9, which sits above engaging, and weak vertical preferences and the default fire sit at the bottom. The turn weight at 9 is below the strong vertical weight at 12, so I'm checking whether that ordering is actually what I want.

I'm settling on a scale for attraction so a single target gives a moderate signal around 0.5 to 0.8, while several targets stay near 1.2 and don't tip into the strong range. The strong threshold sits around 0.95, which blocking and dodging easily clear at magnitude 4 to 5. I'm also wondering whether an enemy in the same row should soften the attraction term.

I'm weighing whether to skip diver signals in the engagement term entirely, since divers drift slowly and chasing them could waste many steps. I'm considering a combined linear term over the diver and enemy channels in the horizontal unit.

I'm working out how to weight the left and right enemy and diver signals, counting enemies at twice the weight of divers and bullets. A single diver or bullet would then give about half the response of an enemy, so I'm checking whether the thresholds for firing versus turning still separate those cases cleanly.

I'm checking whether diver signals should drive engagement at all, since a diver in my row would make me hold position and turn toward it while it drifts slowly, and I'm weighing whether to count divers at half the weight of enemies in the horizontal term instead.

I'm thinking about whether trail positions can tell which way a diver is moving, since that would stop me from waiting on one that drifts away. Counting trails alone won't work, because a trail can sit on either side of an object, so I'd need to weight them by column to tell the two cases apart.

I'm defining a surfacing unit that should fire only when the oxygen is low and a diver sits in the far column, using a weighted difference of the two column signals with a negative bias.

I'm setting the row-1 gate on the up logit: a penalty of 15 when row 1 holds two hazard cells, and a bonus of 25 when the surfacing unit fires.

I'm now adding the "block above" signal as a penalty on the up logit too, so a blocked row above drops the up preference even while surfacing. That leaves a little weight on going down when something sits above, which I'm treating as acceptable for dodging. I'm also checking what happens at row 0 when the surfacing unit is still active before the first dive.

I'm checking the row-0 case: the horizontal signal there reads as a fire-right pull, but the strong down term outweighs it. I'll keep the decode threshold at 0.76 for firing.

I'm defining the horizontal signal per row as a weighted difference of enemy and diver counts on the left and right, plus a facing term and a gate that depends on bullets in the row. I'm still checking whether the left and right regions cleanly exclude the cell the sub occupies.

I'm checking the rest of the horizontal signal: the left and right regions only count enemies and divers, and the sub's own cells are excluded from them. I'm also noting that trail cells in the sub's row can leak into the facing term when an object is close by. The gate's weight on the bullet channel depends on the column, so I need to keep that dependence in mind.

I'm settling the horizontal signal as a weighted difference of left and right enemy and diver counts, with a facing term and a bullet-dependent gate, and I'm checking the scaled values against the fire and turn thresholds.

I'm now defining the layer-2 decoding units on the summed horizontal signal, shifted by 7 so a neutral board sits at zero. Each one is a tanh threshold: one for firing left or right at the stronger cutoff, and one for turning at the weaker cutoff, so the fire and turn cases separate cleanly.

I'm settling the gain for the decoding units at 30, since that puts the gaps between the fire and turn cutoffs well into saturation. I'm also checking whether the absent rows add noise to the horizontal sum, because their trail-based facing term could push the total off-center.

I'm now setting up the vertical signal for rows 1 through 8, combining an attraction term, blocking, dodging, and the same row gate used for the horizontal unit.

I'm setting the vertical signal for each row as a sum over the other rows, with objects above pushing up and objects below pulling down, scaled by distance. A single target one row away gives a fairly weak pull, so it stays below the firing range.

I'm now adding a blocking term for the vertical signal: danger cells in the row above push the sub's vertical preference toward avoiding them, and I'm also checking the thresholds for weak moves so a single distant target still registers.

I'm finishing the blocking term for the vertical signal: danger cells in the row above and below push the preference away from them, with heavier weight on the central columns. I'm also adding a slight downward bias at row 1 so the sub doesn't settle there, and I'm noting that row 8 can't go any lower, so the blocking term below it doesn't apply.

I'm checking whether bullets can reach the sub before it moves: the danger channel covers enemy fish, enemy subs, and enemy bullets, so I'll keep those cells blocked. A bullet in column 2 moving right would reach column 5 in about three steps, while enemies only advance every few steps, so entering the front column still needs a timing check.

I'm working out how the dodge term in the vertical signal should count enemy bullets and divers in the sub's own row, close to its columns. Bullets push toward moving away, and divers cancel that out.

I'm checking the dodge term for the vertical signal: enemy bullets in the sub's own row push it away from them while divers cancel that out. Downward dodging makes sense near the top, upward near the bottom, and it should beat the horizontal engagement when a bullet and an enemy share the row.

I'm checking the case where the sub is in the row with an enemy sub approaching: the left-right signal cancels out, so the choice falls to vertical attraction, which would be a poor outcome. I'm also looking at turning when the enemy sits in the cell just ahead.

I'm checking the surfacing case now: when the surface unit is active, the upward pull is strong, but if something blocks the row above, that block should push the sub back down, so I'm subtracting part of the upward weight to let the down choice win.

I'm working out how the up pull should behave when surfacing: the surface unit adds a strong upward term, and I need a row-1 penalty that cancels it. Checking whether a layer-2 unit that fires on row 1 without the surface signal can supply that cancellation.

I'm settling the up-pull penalty for row 1: a layer-2 unit that is positive only when the sub is on row 1 and not surfacing, subtracted from the up logit. I'm also checking whether the oxygen bar can reliably signal surfacing from the state.

I'm checking the edge case where the diver count reaches 6 during a surfacing step: the counter gets set to 0 and then decremented, which would leave it at -1 and make the diver bar render as empty. I'm treating that as a likely bug in the environment's logic, and I'm noting that pickups would keep counting from there.

I'm noting that sticky actions could cause trouble at row 1: a 10% chance of repeating a prior "up" move would take the sub to the surface and end the episode if no divers are collected.

I'm weighing the danger of moving up into row 1: if the sub enters row 1 by an "up" move and sticky actions repeat it, the sub surfaces and the episode ends whenever no divers are on board. I'll check whether the diver bar can tell me that divers are present, so I can allow the move only when a diver is aboard and an accidental surface just costs a diver.

I'm checking whether the oxygen bar can tell me if divers are aboard: once oxygen reaches 180, column 8 is lit no matter what, so that signal is ambiguous at high oxygen.

I'm ruling out row 1 for the up move, since sticky actions could repeat it and surface the sub with no divers aboard, so I'll only allow that row when surfacing. For the other moves, sticky repeats could push the sub an extra row or turn it further, but I'll accept that since the blocking check only covers adjacent rows.

I'm dropping the row-1 attraction so the sub doesn't climb toward targets it can't safely reach, and I'll penalize up moves from row 2 when not surfacing, so the sub stays put there.

I'm checking the row-0 case now: diving from the surface into row 1 skips the blocking check, since the vertical signal is gated off at row 0, so an enemy in row 1 could be hit. I'm considering a separate danger unit for row 1, columns 4 through 7, to cover that move.

I'm adding a row-1 danger unit for the case where the sub dives from the surface, since the vertical signal is gated off there and nothing else would catch an enemy in that row. Then I'm giving row 1 only a weak downward bias, so a blocked row below pushes the sub toward up and the row-1/row-2 penalty holds it in place.

I'm checking whether divers can reach the sub's front column without being missed: they cross the whole screen, so a diver coming from either side passes through the front cell and gets picked up, and bullets don't affect them.

I'm weighing a diver-friendly version of the engagement term, where divers get a weaker pull than enemies so the sub doesn't sit idle waiting for them to cross the screen.

I'm dropping the diver-only signal at 0.8 into the turn range, since a sub that keeps turning toward a diver would drift off center, and I'm looking for thresholds that keep the fire, turn, and vertical regions separate for enemies, divers, and bullets.

I'm testing the blend weights for the diver and enemy signals: enemies count fully, divers half, and a small shared bias on top. Setting the bias to 0.2 keeps diver-only inputs below the turn threshold while mismatched enemies stay above it, so the sub won't drift toward divers.

I'm checking the diver-in-own-row case: attraction only counts other rows, so a diver in my row won't hold me in place while I drift toward other targets. I'll need to add something for my own row so I stay put while a diver approaches.

I'm considering whether a diver-only signal can be given a "stay" effect in the sub's own row, since right now divers in my row don't hold me in place and attraction from other rows pulls me away before they arrive.

I'm weighing whether the diver-and-enemy signal can be split cleanly: at 0.2 bias, a diver in my row lands in the "none" region, so vertical moves stay allowed. Then I'd lose the hold on approaching divers, and the wait for one to cross the screen could run around 30 steps. I'm leaning toward accepting that cost and keeping one shared engagement term.

I'm settling on a single weight of 0.3 for the diver-and-enemy signal rather than splitting them, since trying to separate divers from enemies with a trail filter isn't worth the extra complexity. Then I'm checking the facing convention: with F=+1 meaning facing left, an enemy on the left gives a positive value and fires, while an enemy on the left with F=-1 gives a turn-left value.

I'm working out the facing term for each row: the left-minus-right weighting per column, with the bias scaled by the facing sign, so that a matched enemy comes out as a fire signal and a mismatched one as a turn.

I'm checking the mirrored case, where the sub sits at columns 3 and 4 and the facing sign flips the sign of the fire-versus-turn value.

I'm checking how the sub's position affects the facing term: the sign flips when the sub sits outside columns 4 through 7. To keep it centered, I'm considering a recentering term that nudges the sub right when its front is near the left edge and left when it's near the right edge, though it could interact with the turn logic and cause oscillation.

I'm checking the recentering idea: with the sub at columns 3 and 4 facing left, a fire signal competes with a right move, so I'd want the recenter logit set between them. Moving right from there takes two presses to get the back cell clear of column 3, so the nudge is cheap. I'm also counting how many layer-one units that would take, starting with the vertical and horizontal row detectors.

I'm settling on the diver-and-enemy weights and checking the rest of the layer-one units, which comes to 21 with 11 spare.

Now I'm working out the vertical signal in layer two, combining the row sums into a sub-position value so the up and down units can be built from it.

Still working out the vertical signal: these are the four up and down units with their thresholds on the summed row values, and I'm checking that row 0 correctly drives the strong down unit.

Now I'm checking the vertical gain and thresholds: a weak band for attraction and a strong band for dodging or blocking, with the tanh outputs staying clear of each other.

Still working out the decoded fire and turn signals in layer two, using the H-sum offsets and gains to set the thresholds.

I'm checking the turn-right indicator now: it comes from the turn-right signal minus the fire signal, halved, and at row 0 with the sub's sum at -8 the fire unit wins out.

Now I'm writing out the layer-two signals for the sub's front-row status and the row 0 and row 1 checks, using tanh units with a fixed gain so they stay sharp at the thresholds.

I'm writing out the layer-two logits now, starting from the fire unit's base value of 3 plus a term scaled by the fire indicator.

Still writing the logit formulas...

Checking the surfing case when a diver-or-bullet block sits above: the down logit gets 12 from the strong dive signal, minus 5 from the surfacing term, which leaves 7 and still wins over the others.

I'm checking the surfacing case: with a diver or bullet blocking above, the down logit still wins, though moving down risks running into danger below, which the vertical signal should cancel. Staying put would be better than diving, but the extra two steps are affordable with the oxygen margin. For normal hunting with an enemy three rows up, the weak up signal gives up a logit of 5 against fire at 3, so up should dominate.

I'm settling the weights so the vertical signal dominates: weak attraction at 8, default fire at 3, and fire on engagement at 11. Strong vertical dodging gets 16, surfacing 18, and the row-1-and-2 penalty -35. Next I'm recomputing the left and right turn logits from these numbers.

I'm setting the weights for the right-turn logit and the vertical up and down logits, with fire at 11 and the strong vertical dodge at 16, so an engaged fire still beats a weak vertical move.

Checking the down-versus-fire cases at row 0 and the surfacing edge: a diver-blocked dive still lands on fire, and when nothing blocks it the down signal wins.

I'm checking whether the sub can get stuck oscillating: after diving from row 0 to row 1, the attraction toward targets below pushes it back down, and a blocker below turns the up move into a penalty, so it should just stay put. Near the bottom edge, a down press does nothing, so the attraction there only ever points up or stays neutral. When a blocker sits above, the strong down signal outweighs the attraction, but once the sub has moved down, the target is two rows up and the logic could flip back.

I'm settling the blocking term: a blocker above shouldn't push the sub down by a fixed amount, since that just makes it oscillate. I'm thinking about folding it into the log so that staying put wins when something is above.

I'm still working through the blocking term: rather than giving each row its own unit, which would take fourteen units and I only have eleven spare, I'm looking at merging the blocked-above and blocked-below checks into a single gated sum.

I'm working out how to fit the blocking checks into the unit budget: the per-row blocked-above and blocked-below terms would need fourteen units, but dropping the recentering pair frees enough room. I'm settling on a separate blocked-below unit for each row from 0 to 7, so the row-0 check is folded into that set.

I'm trimming the unit budget: dropping the recentering pair frees room, but I'm still one unit over, so I'm considering whether row 1's vertical unit can go, since it would push a surfacing sub back down.

I'm now dropping the penalty on the up move that comes from the surfing-and-blocked-above check, since a separate per-row blocking unit can handle that case. That way a surfacing sub at row 1 can still move up without getting stuck.

I'm trying to drop the penalty on the up move that comes from the blocked-above check, since a separate per-row blocking unit can handle it, and then I'm checking whether row 1 still needs its own vertical unit once the offset is applied.

I'm dropping the row-1 horizontal unit as well, since engagement there would just keep the sub at row 1 while the strong down signal already wins. That leaves seven units for the horizontal terms on rows 2 through 8, with the offset folded into the sum.

I'm trimming the unit count: the horizontal units can drop to rows 2 through 8, and the blocked-below and blocked-above gates take up most of what's left. Counting it all out, the total comes to 31 units, so there's one spare, and the recentering pair still needs a combined signed unit to fit.

I'm settling the layer-two decode: a single signed recentering unit feeds two readouts, one for pressing right and one for pressing left, which brings the total to 32 units. I still need to check whether a blocker below row 1 could keep the sub from diving past row 1.

I'm mapping out the input indices for each row's channel-0 cells, with row r starting at r*60 and each cell taking six channels, so the gate for the blocking units can read the right positions.

Still writing the gate indices for the per-row units...

I'm now wiring the per-row vertical units for rows 2 through 8, giving each a bias and a fixed attraction weight toward targets above and below, with the dodge terms split by channel so a diver cancels a bullet while a lone diver still pushes the sub upward.

I'm stuck on the dodge term: a single linear weight can't express "bullet means ch1 without ch2," since a diver alone in a cell would push the sub up just as hard as a bullet pushes it down. I'm considering whether any weighting of the two channels can separate them, or whether the dodge needs a different encoding.

I'm now treating the dodge as its own gated unit, since a single linear weight can't tell a bullet from a diver. A bullet is an enemy-channel signal without a matching diver signal, so I'm weighting the enemy channel positive and the diver channel negative and putting a threshold on the sum. That should keep a lone diver from pushing the sub up.

I'm trying to make the blocking flags fit: one option is merging the up-blocked and down-blocked checks into a single signed unit per row, though a case where both directions are blocked would cancel to zero and go unflagged. I'm also weighing whether the freed units can go to the dodge term instead, since a thresholded unit might separate bullets from divers where a linear weight cannot.

I'm reworking the blocking flags into one signed unit per row, so up-blocked and down-blocked come out as opposite signs, which frees up a few units for the dodge term.

I'm checking the unit count again: the vertical and horizontal terms take fourteen units, and the blocking flags take nine more, so dropping the recentering unit gets me to thirty-two. For the dodge term I'm trying a gated unit per row for rows 2 through 8, where a bullet gives a strong positive signal and a diver cancels it, though a row with several bullets could push the baseline a little off.

I'm settling the dodge unit for rows 2 through 8: a gated threshold per row, with bullets weighted positive and divers negative, so a lone bullet crosses the threshold while a diver cancels it. I'm setting the gate high enough that absent rows stay near the floor, then summing the per-row dodge outputs and reading the result in the second layer. Direction will prefer down on rows 2 through 7 and up on row 8, with the row-1 penalty keeping the sub from surfacing into the blocked zone.

I'm settling the dodge direction now: each row's dodge unit feeds a shared signal that pushes the sub down on rows 2 through 7, and I'm giving row 8 its own indicator so the sub goes up there instead, since pressing down at row 8 would do nothing.

I'm settling the logit contributions now: the dodge-other signal pushes the sub down with a strong weight, the row-8 dodge indicator pushes it up, and a weaker up term covers the case where down is blocked. Blocking flags subtract heavily from whichever direction they cover, and I'm checking that the dodge weight beats the engage term.

Still working out the dive term: the attraction values could push past what I want, so I'm checking whether the dive indicator at rows 0 and 1 can rely on the shifted sum alone.

I'm settling the final layout at 32 units: vertical units for rows 2 through 8, horizontal units for the same rows, nine blocking flags for rows 0 through 8, seven dodge units for rows 2 through 8, plus the shared signal and the row 1-2 penalty. The recentering unit is dropped.

Still working out the blocking units: I'm setting each row's signed flag so that danger above and below come out with opposite signs, and I'm checking the magnitudes so a single blocked cell crosses the decode threshold cleanly.

I'm settling the blocking units now: a G of 16 keeps absent rows well clear of the threshold, and the layer-two decode puts the up and down blocks on opposite sides of 0.5.

I'm checking the blocking unit's margins: an absent row sits near -16 plus or minus 8, so it stays essentially at -1 and doesn't leak into the decode. I'm also deciding that the blocking check should only look at enemies in the four columns around the sub's front, since an enemy at column 5 with the sub entering at column 6 could be a collision or could be drifting away. Finally, I'm weighing whether a bullet a few columns back and moving toward the sub should already trigger a dodge, since the dodge unit covers that range.

I'm narrowing the blocking check to a few columns on each side of the sub's front, and I'm tuning the dodge's near-cell window so a bullet close enough to hit in two steps still triggers an immediate move.

I'm checking which bullets could reach the sub from the back cell as well as the front, so the dodge window needs to include columns 5 and 6 on each side.

I'm checking the bullet-versus-diver case in each row: a bullet and a diver in the same row cancel out, so that row won't trigger a dodge.

I'm finishing the attraction term now: each target counts with a distance weight, and I'm working out the falloff values for rows 2 through 8 so that nearby targets dominate without pushing the sum past the thresholds.

I'm writing the value-term decode next: a weak up signal and a weak down signal from the summed attraction units, with the down one also covering the dive at rows 0 and 1.

I'm now writing out the horizontal decode: a forward-fire unit that fires when the horizontal sum is strongly positive, a shifted version that covers the stay-and-fire case at rows 0 and 1, and a threshold for the turn-around signal.

I'm writing out the blocking decode next, which uses a sum over the nine blocking units and shifts it by eight before comparing against the threshold.

Writing the output decoding units now, checking the threshold margins on the dodge and attraction signals.

Still working out the sub's decode formulas, now on the fire-and-dodge signals in the same layer.

I'm writing out the logit contributions for each output action: no-op stays at zero, fire combines the two forward-fire indicators, left and right come from the horizontal threshold pairs, and up draws on the vertical signals, the shared dodge indicator, and the row-1-and-2 penalty.

Still working out the down and up logits for the surfacing and blocking cases, checking that surfacing while dodging comes out as up, and that a blocked surface pushes the sub into staying put.

I'm checking the row-1 surfacing case: the dive signal stays strong while the sub is still diving, and a dodge signal at row 1 flips it back up, with the penalty term keeping it from surfacing early.

I'm checking the row-0 and row-1 case: with the sub at column 1 and the horizontal sum at -7, the turn signal and the forward-fire signal both saturate, so right cancels to zero and fire sits at 11. The vertical signals also saturate, so down wins there at 24. Engagement still leaks about 5% into vertical moves, since fire at 11 only edges out the weak vertical logit of 8.

I'm raising the engagement fire logit to 13 so it beats the vertical signal at 8, and setting the turn logit to 14. For the default case with nothing engaged, I'm lowering the fire baseline to 6 so random up and down moves stop leaking in.

I'm resetting the logit weights so the idle fire baseline is 6, engaged fire reaches 14, and the vertical and dodge terms scale up to match, with the blocking and penalty terms strong enough to override them.

Still tuning the logits: engaged fire at 16 keeps the vertical leak small, so I'm raising the dodge and surfacing weights to 24 to stop them losing to firing.

Still tuning the logits: I'm checking the surf-and-dodge case, where the up logit reaches 38 and clearly wins.

Working through the edge cases now: a blocked dive at row 0 drops to firing, and a dodge at row 2 can still end in a hit. Checking the idle baseline, the fire logit of 6 wins about 98.8% of the time when nothing is engaged.

I'm checking the edge cases now: a weak vertical move into a blocked row still gets vetoed, and engaged firing takes priority over drifting. I'm also confirming that repeated fire presses during the shot cooldown do nothing, so there's no movement cost there.

I'm checking the sticky-action risk for moving up into row 2: if the repeated up press fires, the sub lands in row 1, and a second repeat could take it all the way to the surface, so I'm estimating that at about 1% and deciding whether to ban that move.

I'm checking the last edge cases: a sticky repeat of an up move could carry the sub from row 2 to the surface, which I'm weighing against the reduced target coverage that banning it would cause. Engagement looks fine since the bullets outrun the enemies, and horizontal drift only happens when the sub turns.

I'm now checking whether the horizontal recentering can be folded into the existing turn units by widening their column range from 4..7 to 3..8, so that drift at the edges still gets corrected.

I'm checking whether widening the column range is safe: the horizontal difference b - f holds for any centering point, so the turn signal still works if I extend the weights across all columns, though I need to make sure stray trails in the own row don't leak into it.

I'm checking whether widening the column range to 3..8 would help, but the extra columns pull in trails from enemies just outside the front, which would throw off the turn signal right when engagement matters most.

I'm checking whether the enemy trails at columns 3 and 8 would leak into the turn signal if I widened the range, and they would, so I'm keeping the 4 to 7 window. I'm also mapping the left and right regions onto the sub's columns, where a sub sitting at 4 or 5 or at 6 or 7 leaves the object at column 5 unaccounted for.

I'm checking whether the trails of enemies approaching from the sides would land inside the 4 to 7 window, and for the cases I'm testing they stay outside it, so the turn signal holds.

I'm checking the policy against the enemies' trails in the own row: approaching enemies stay outside the 4 to 7 window, so the turn signal doesn't pick them up. An enemy behind us moving away still gets caught by a turn and a shot. When both sides carry a signal, the horizontal logits cancel and the sub falls back to vertical movement or idle firing.

I'm running through the enemy-approach cases: approaching enemies have their trails outside the 4 to 7 window, so the turn signal ignores them, while an enemy that has already passed us still gets caught by a turn and a shot. I'm also checking that vertical moves through rows with enemies are fine, since blocking keeps the center columns clear.

I'm checking the bullet case: an enemy's shot starts in the sub's own cell, so the horizontal signal ignores it at first, and once it moves ahead it only matters if it lands in the window. I'm also confirming that after a dodge, a bullet still in flight could make me dodge again when I come back up.

I'm checking the high-position and oxygen timing: the sub needs to surface when oxygen drops below 60, and from row 8 that takes about eight steps, a little longer with blocking delays.

I'm checking the oxygen threshold for surfacing: once oxygen is at 180 or above, the surface-col signal drops out, so the surfacing logic only triggers when a diver is aboard and oxygen falls under 60. I'm also setting up the horizontal gating term for the H units, where the ch0 cells contribute a fixed amount when the row is present.

I'm checking the idle-state baseline for the horizontal signal, where a constant of -12 keeps the horizontal logit strongly negative when no enemy is in the window. I'm also settling the constants for the other units, with the bullet term at -16 and the vertical term at -12, and I'm confirming that the cell index helper maps row, column, and channel into the flat layout.

Writing the weight index helpers for each unit now...

Continuing the per-row weights for the vertical units, I'm applying the distance decay values to each row above and below, with negative signs for rows below and positive for rows above.

Still filling in the per-unit weights for the vertical units...

Still filling in the per-unit weights for the vertical units...

I'm mapping out the rest of the bullet weights for the vertical units, starting with the ch2 entries across each row, and I'm settling on scalar fills so each row-and-unit pair takes one line.

Still filling in the weight matrix entries for the horizontal units, starting with the ch0 gating cells across each row.

Still filling in the per-cell weights for the horizontal units...

Still filling in the horizontal unit weights...

Still filling in the weight entries for the bullet-danger units...

Still filling in the weight entries for the bullet-danger units...

Still filling in the weight entries for the Y units, covering the bullet and enemy channels across each row...

Still filling in the weight entries for the Y units and the bullet-danger units, and now moving on to the special sf and R12 units and the layer-2 weights...

Still filling in the weight entries for the output layer, setting the H2 units' weights from the H1 units and their biases.

Still filling in the output-layer weights, now checking the threshold for the VUw unit at a typical present state.

Checking the value-head thresholds: with V_sub - 3 and the VDs unit at -30 V_sub - 28.5, the cutoff sits near 0.95, and the forward and backward pairs for H and B work out to the expected margins.

Still working out the layer-3 logits, rewriting the sigmoid-style terms as scaled halves of the hidden outputs.

Still filling in the output-layer weights for the fire, left, and right logits...

Still filling in the layer-3 output weights for the up logit.

Still filling in the output-layer weights for the down logit.

Checking the threshold for the value-head units: with V_sub at -1 the VDs unit only reaches about 0.905, so I'll lower its cutoff to 0.92 to get a saturated value near 0.98.

I'm lowering the gain on the VDs unit to 60 with a threshold of 0.93, so at V_sub = -1 it saturates near 1, and when attraction is present it drops back below the cutoff.

Still checking the down-logit at row 0 and the R12 unit there.

Still checking the surfacing and sonar signals against the row-2 weights, and now I'm turning to the case where the column-8 cell is on and column 2 is off.

Still checking the dodge case at row 5, working through the down-logit terms as the diver signal turns on.

Checking the down-move at row 5 with the diver signal on: the down logit comes out around 24 to 34, while the up logit lands near 14 to 24, so moving down still wins. Now I'm verifying that the bullet penalty holds when surfacing is blocked.

Still checking the horizontal logits against the H values, working through the facing-left case where the enemy sits on the left side.

Checking the weights for the vertical value unit with a single distant target, the output comes out near 0.98, so it still fires as expected. Then I'm confirming the rows 0 and 1 case, where the value unit saturates at -1, and that the engagement-plus-surfacing logit wins over firing.

Still checking the bullet-danger term for the unit above, where the danger signal from the row above is what blocks upward movement.

Now I'm checking the boundary-danger weights, where units look at the row just above and just below for bullets, and the indices step through each row's column range.

Still setting the weights for the bullet-danger units, now the ones for rows 2 and 3, where the danger above and below each row feeds in with opposite signs.

Still filling in the weight entries for the Y units, covering the ch0 gating and the enemy channel across each row...

Still filling in the weight entries for the Y units, covering the ch0 gating and the enemy and bullet channels across rows 2 through 8...

Still filling in the weight entries for the H units, working through the channel-0 gating and the left-facing and right-facing terms row by row.

Still filling in the weight entries for the H units, now the ones for rows 5 and onward.

Still filling in the weight entries for the V units, starting with the channel-2 attraction rows for each distance step...

Still checking the signs on the bullet-danger weights, and I'm wondering whether two equal attractions pulling up and down might leave the sub idling between the two targets.

I'm now weighing the tie-breaker problem: two equal pulls from targets above and below cancel out and leave the sub idle. One option is to weight nearer targets much more heavily and lower the weak-signal threshold so that distant targets still register.

I'm tuning the direction preference so that upward targets count a bit more than downward ones, which breaks ties between equal pulls. I'll lower the weak-signal threshold to about 0.05 and raise the gain to 60, so distant targets still register while two equal pulls no longer cancel out completely.

Still checking the up-versus-down bias, and I'm setting the value-unit gains and thresholds so that a tie between two equally distant targets doesn't cancel out completely.

Still working out the value-head weights for the up and down sides, scaling the down-direction pull by 0.85 so the tie between equal targets breaks toward going up.

Still filling in the weights for the value units, now the rows at distances three and four from the target above and below.

Still setting the value-unit weights for the rows further away, with the down-direction pulls scaled by 0.85 so up wins ties, and checking that a strong pull from several targets stays under the saturation cutoff.

Still working through the value-unit weights for the down direction, checking that the 0.85 scaling and threshold leave the attractions from several targets under the saturation cutoff.

Still filling in the weights for the output units, now the bias and the connections for the horizontal and bullet groups.

Still filling in the weight entries for the output units, now the fire and left-facing terms, which carry the bias values alongside them.

Still assigning the output weights: the right-moving unit gets its opposing terms from the left-facing and right-facing signals, and the up and down units pick up their bullet-danger and attraction contributions across several rows.

Checking the engaged case at row 8: the up and down weak signals each come out to 10, so the fire unit at 16 stays ahead, and the left turn sits at 17 against fire's 6.

Now I'm writing up the edits for the design brief and checking the remaining tie-breaking and sign details.

Keeping this short, I'm just tidying up the edits block and the design brief now.

