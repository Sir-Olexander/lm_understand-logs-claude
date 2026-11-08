I'm noticing the model passes with near-certainty while dame points remain, since the best pass logit is now -5 and every dame move sits below that. My guess is they're penalized either by the uncertainty term or because they really are self-atari, with the dame sitting between groups that have few liberties. I want to check a few final positions, like games 70, 75, 123 and 47, to see why those dame points score so poorly.

I'm still weighing whether those dame moves are penalized by the uncertainty term or whether they really are self-atari, with each dame point sitting between groups that have few liberties.

