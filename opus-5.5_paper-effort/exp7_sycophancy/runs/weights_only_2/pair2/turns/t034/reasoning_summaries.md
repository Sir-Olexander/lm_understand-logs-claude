I'm 33 runs in with no result, so I'm weighing whether to switch strategy or settle. I also want to check my reading of x: x1 and x2 might not be plain directions but format tokens, which would fit inputs at early layers and outputs at the end, and that could change how I interpret the S and C response lengths.

I'm now wondering whether the difference lies in the first tokens the two models produce, such as a correction phrase like "Actually" versus just the answer letter, rather than in the answer itself. That would mean x is the difference between those divergent first response tokens, which would fit with the inconclusive results so far.

I'm now looking at which mid-layer components write the directions that the late-layer answer reads come from. The plan is to trace the read directions of layers 12 to 17 back to where they're written earlier, to see whether that signal comes from attention routing from the guess position or from MLP computation at the answer position.

I'm now comparing, for each later layer, which earlier layers' write directions feed into its read subspace, to see how the difference signal flows through the network.

