I'm noting that v3 fills eyes often: 714 eye fills, 61 of them self-atari, which confirms the bug with the eye exception and low SA0. I also want to check the eye fills that aren't self-atari, where the merged group still has two or more liberties, since an eye unit should still get the -40 penalty when all its neighbors have zero liberties.

I'm checking whether the eye exception can be made safe. The lower-bound detection means a group with three liberties can look like it has two, so the exception could leave real eyes unprotected. I'm leaning toward reverting to always penalizing eye fills and finding another way to allow the false-eye escape, since the obvious linear fix doesn't work.

I'm working out a way to keep the eye protection while still allowing the false-eye escape, using a term that scales with each neighbor's liberty count. The idea is to subtract the neighbors' liberty signals so an eye fill is suppressed only when a neighboring group is in atari, though this still penalizes some cases conservatively.

I'm checking whether the atari-based suppression holds up when the atari judgment itself is wrong, since a false atari would let an eye fill through on a group that isn't actually in danger. The penalty for the unit would then have to be strong enough to cover that case.

I'm weighing whether to drop the eye-point suppression entirely and always penalize eye fills, since the atari-based exception could let fills through when the atari judgment is a false positive and the eye is real. Missing some false-eye escapes may be the safer trade, though several of the sampled missed saves were exactly those connections.

I'm weighing whether to drop the eye suppression altogether, since a true false-eye connection would still be penalized heavily under the current weights and rarely gets played anyway. Accepting never to fill eyes would mean reverting SA0 to 1.5, and I still need to check how the other weight changes from v2 to v3 affect this.

I'm reverting the eye and SA0 weights and keeping the others, since the other v2-to-v3 changes might also be hurting. Pass 4 still has about 321 self-atari moves outside eye fills, far more than the 114 in the baseline, so I want to sort those self-ataris by their number of eye neighbors and by the liberties of the adjacent group to see where they come from.

