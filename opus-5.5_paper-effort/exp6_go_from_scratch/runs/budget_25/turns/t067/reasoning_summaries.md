I'm checking a move at (2,8) with about 0.9995 probability, which would put our own two stones in atari. Looking at the board, that point touches an opponent stone above, one of ours below, and an opponent stone to the left.

Now I'm checking whether the group at (2,8) and (3,8) has only one liberty, which would make it a self-atari of size 2. I'm also wondering why we'd play there, since the opponent group above and to the left looks like it might be a false-positive capture target. Then I'm sizing up the same kind of self-atari at (8,3) in another game.

I'm checking the big O group around (8,2) as a possible false-positive capture target, since it has only a couple of obvious liberties but probably many more further away. I'm also thinking that this same pattern could explain the repeated self-ataris in games 41 and 48, and that false-positive captures on large groups seem to be driving the snapback cycle and repetition losses.

Checking the pair at (2,7) and (1,8): (2,7) has two empty neighbors, (1,7) and (2,8), so it isn't in atari, and (1,8) has three empty neighbors. That means neither opponent stone looks like a true atari target here, which leaves me puzzled about why (2,8) scored at 0.9995.

I'm checking whether (2,8) might be in atari ourselves, but (3,8) has two liberties, so it isn't. I'm wondering what else could give it such a high score, so I'm looking at the positional term, where the stones within a few points of (2,8) on each side might be pushing the value up.

I'm working out the rest of the score for (2,8): the stones it's close to push the value up, but the net comes to only about 1.2 once the penalties are subtracted, so the move looks like a marginal logit rather than a clear winner. Passing scores far lower, around -8.

I'm working through the penalty for (2,8): the unsuppressed U0 term would push its logit to about -15, below passing at -8, yet the move still scores 0.9995. So some large bonus must be offsetting it, and I'm now checking whether an escape term is responsible for that.

Checking whether the 0.9995 score comes from the capture term: the two neighboring opponent stones each have two or more empty neighbors, so the capture bonus should be zero. Something else must be pushing the value up, so I'm going to run a positions pass on this position and a few other repetition-start positions to see the activations directly.

