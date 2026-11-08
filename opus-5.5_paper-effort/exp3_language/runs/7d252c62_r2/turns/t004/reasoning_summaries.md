Both L0H0 and L0H4 write into nearly the same 2D output subspace, since their fractions for the same vector are around 0.8 and 0.9, and both look fairly uniform. Next, I'm checking whether the L1 heads' bias-driven preferred key direction, with norm 2 to 3, lies mostly in that shared subspace.

I'm now asking whether the L0H4 output is just a running average of values, so the L1 heads would be keying on a position-dependent mean, and I'm looking at whether a constant component across tokens would make that signal uninformative. I'm also wondering if the large singular value of L0H4 (around 13.9, versus about 1 for the others) means its output dominates the residual stream after layer norm.

I'm checking whether L0H4's output writes a large constant value into residual dimensions 123 and 21, since those dimensions carry big LayerNorm biases. If so, it could act as a massive-activation direction that L1 keys read to produce a query-independent position preference, though a constant component might cancel out and leave no differential signal.

I'm working out whether L0H4's output varies across key positions, since a constant component would give no gradient signal for the L1 key weights. The softmax gradient sums to zero over keys, so only a content-dependent running average could be what L1 heads actually use to prefer some keys over others.

I'm weighing whether L0H4's output could carry a position signal, since a running average of positional embeddings would vary smoothly with position and is something the L1 keys could read. That seems roundabout, though, since the positional embeddings are already available to L1 directly. I'm also wondering what a shuffled-text model would need: counts of characters in the current line and some sense of progress through it.

I'm now considering whether the L1 heads could use a shared newline-based signal to tell which line a token belongs to. A uniform L0 head that averages newline occurrences wouldn't cleanly separate tokens within the same line, since its value shifts at every newline. A stronger route is for an L0 head to attend to the most recent newline, letting later heads read off where the current line began.

I'm going to check whether L0H4's value vectors read a token-identity direction or a positional one, by comparing them with the query and key directions of the positional L0 heads like L0H1 and L0H5. I'll also look at the top left singular vectors of L0H4's output to see which residual dimensions, such as 123 and 21, it writes into.

I'm checking which residual dimensions L0H4's output writes into, looking at the largest entries of its left singular vectors, and seeing whether its right singular vectors sit on a single basis-aligned input dimension with a large LayerNorm weight.

