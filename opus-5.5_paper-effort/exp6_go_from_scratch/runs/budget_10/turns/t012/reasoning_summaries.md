I've confirmed that the 78 false-atari stones with no seed in their group, and the 105 at chain distance four or more, account for the errors, while none occur within distance three, so the implementation looks correct. Now I'm turning to the group sizes for the false cases, where most fall into small groups under ten stones, and the seedless ones seem to be isolated stones each with a single liberty.

I'm weighing whether seed coverage could be improved for the seedless cases, and then turning to whether propagation could reach further. Combining a propagation step with the seed computation in the second layer only extends the simple seeds by one step and doesn't add pair-seed reach, so I'm looking at longer-range jumps or a row-wise sweep that propagates along contiguous horizontal runs in one layer.

I'm considering whether a per-point gap penalty could make a single unit compute contiguous own-runs, but the penalties for different distances would couple together, so that doesn't work. I'm also noting that depth is the real bottleneck here, since width is plentiful in the first hidden layer but propagation needs information in later layers. I'll look at where the practical value lies, since groups with few liberties in real games are the attack targets that matter most.

I'm weighing how much a false atari costs in practice: a bonus on a liberty of a large group wastes a move at a critical moment, and for our own groups it could mislead the escape and semi-atari estimates. Improving the seeds looks cheap, since adding the F3 seeds into F is already how A is built, and I'm considering what other seed pattern might help.

I'm considering what the groups without any seeds look like, so I'm going to examine a few concrete examples of them.

