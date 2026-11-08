I'm tracing through a Go example where our group is enclosed by white stones in the top-left corner. Playing (0,2), then (0,0), then (0,1) and (1,1) fills in our own eye space, so the group ends up with no liberties left.

I'm checking whether the (1,0) stone really is certified, since its two liberties are (0,0) and (1,1). That seems to contradict a false atari flag, so I want to look at the activations from a positions pass. I'll pull the move list up to ply 40, just before the (0,0) move.

