The eye check fires at (7,8), since all its neighbors are friendly stones, so it counts as an eye and takes a penalty. But that looks wrong: the neighboring string is in atari, and once (7,8) is played the stones merge into a larger group with its own liberties, so it isn't a real eye.

I'm settling on suppressing the eye penalty whenever an adjacent friendly string is in atari, since filling it there connects to a healthy group and saves it. I'd also apply the same suppression to the territory term, and then move on to checking the second example's escape value against the capture.

