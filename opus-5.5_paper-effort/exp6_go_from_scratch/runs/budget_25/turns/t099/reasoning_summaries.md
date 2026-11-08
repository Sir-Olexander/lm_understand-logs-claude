I'm puzzled that the move at about 1.0 probability is a self-atari, since the capture bonus is absent and the penalty is suppressed. The position value there must be high enough to outweigh that, or the alternatives are worse. Game 60's corner move at (8,0) has a line-1 value of -6 plus a late adjustment, and it still comes out at 0.99.

I'm checking whether late-game suppression lets a pass-level baseline make the self-atari win over passing, since the positional value might only just beat it. Cases 71 and 16 show the reverse problem, where the bonus and the suppression both fire. That suggests the suppression should only apply when the atari signal is actually present, but the feature at this layer can't tell that apart, so I'm weighing whether the suppression term belongs in the output layer instead.

I'm checking whether a restore term could recover the penalty only when the atari signal is present, but any product of the two terms isn't expressible in this layer. Scaling the capture bonus up to outweigh the penalty would also flood the output with false positives, so the suppression seems to belong somewhere else in the structure.

I'm weighing whether the suppression could be keyed to the same detector as the bonus, but the bonus sits at layer 6 while the suppression would need layer 5, so that doesn't fit. Keying both to the radius-4 detector would just give the R4-only false positives a bonus too, so they'd still get played. The open question is what it costs to suppress only on those R4-only cases, and whether the detector radius can be pushed out to 5 by restructuring layer 5.

I'm now considering whether the suppression detector could sit at a deeper layer, but layer 5 only sees step-4 information at the stone itself, which makes the radius-4 limit hard to escape without changing how the seeds start.

I'm working out whether a single ReLU at layer 1 can compute the step-1 parity signal for a stone, which depends on its own liberty count and on whether an opposing neighbor has at least one or two liberties.

I'm working through a linear form for the step-1 parity signal, where the inner OR over opposing neighbors is an OR of ANDs that a single threshold can't cleanly express. The problem is that stones or empty points outside the group can leak into the sum and fire falsely, so I'd need the neighbor weight to dominate the liberty weight.

I'm trying to find a linear threshold for the step-1 parity signal, where a stone's own liberty count and an adjacent opposing stone's liberties both need to count. The trouble is that empty or own neighbors around an opposing stone can add enough weight to fire the unit falsely, so the neighbor weight would have to outweigh those contributions.

I'm weighing a few ways to keep the suppression from firing on the radius-4 false positives, but every linear placement I try either leaks the neighbor's empty points into the sum or needs a contradictory weight ordering. So radius 4 looks like the ceiling for this detector. Improving the coloring might cut the false positives, but collisions aren't the main source; the radius is.

I'm checking whether the opposing group's size and liberty count could serve as a signal: the walk counts for huge, well-liberated groups should be large on the other parity channel too, while a true small atari group would show little there. Adding a penalty on that other channel to the suppression unit might separate the two cases.

I'm considering whether the opposing group's other-channel walk sum could serve as an extra penalty in the suppression unit, so that big, well-liberated groups get separated from true single-stone atari cases. For a true atari, that sum stays small, but for larger groups it grows with the walk counts from their seeds.

I'm checking whether the opposing group's other-channel walk sum, scaled down by a small weight, could separate huge well-liberated groups from true single-stone atari, since those big groups would produce large values there while a real atari stays near one.

I'm now testing whether the other channel's walk sum could act as a size signal, so that a huge, well-liberated opposing group produces a large value that dampens the suppression while a true single-stone atari stays near one. The catch is that this quantity is an activation-style computation I'd be carrying out by hand, so I'm weighing whether it's a legitimate way to check the feature design at all.

I'm going to stop trying to compute the walk counts by hand, since that would amount to reproducing the network's activations. Instead I'll use the positions pass to read off the activations for true atari stones and false positives, and pick the weight from those measurements.

I'm going to pull the activation value for each opposing stone from the pass 11 output, since that gives the walk-count sum without computing it by hand. Then I'll compare those values for true atari stones against the false positives and see where a threshold can separate them.

