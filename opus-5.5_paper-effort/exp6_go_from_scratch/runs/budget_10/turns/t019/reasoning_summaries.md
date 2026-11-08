Checking how often Option A raises a false atari: about 2% of stones with two liberties and under 0.5% with three or more. The hash variants do little better, so the failures seem to come from propagation reach beyond distance 3 rather than from the seeds. Next I'm looking at how accurately the two-liberty case is detected, which comes out to about 83%.

Now I'm weighing the false-atari counts against the true-atari ones, which puts roughly 19% of the capture and escape signals in the false category. Since there are far more groups with three or more liberties, even the 0.44% rate adds up. I'm also thinking about how to speed up propagation to extend its reach.

Now I'm weighing whether a few hops of extra reach could be added cheaply, either by giving the simpler seeds one more hop or by using a single unit to propagate faster. The catch is that any shared gating term would also block legitimate contributions from other directions, so per-direction units seem to be the only clean route, and that multiplies the cost by four.

I'm working out whether a horizontal three-point window can give an exact seed for two or more liberties in one unit. Since the three points are in a straight line, their liberties don't overlap, so counting them with a single threshold stays exact.

Now I'm considering a three-stone horizontal unit that can mark a whole straight triple as a seed at once, so that a later layer can spread that flag to each of its stones.

Now I'm checking whether a straight segment of own stones, rather than just a pair, can serve as an exact seed: if every stone in the run is own and the adjacent empty points total at least two, the liberties are distinct and the flag holds for the whole segment.

Now I'm considering whether long straight segments could extend propagation in one step, since a line of own stones with at least two adjacent empty points has distinct liberties. The issue is that the number of segment units grows quickly with length, and real chains bend often, so this only helps along straight runs.

I'm going to measure how much one or two extra hops of reach would help, and whether the failing stones tend to sit far from their seeds along straight lines.

