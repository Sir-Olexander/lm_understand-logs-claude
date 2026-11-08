I'm listing the captures the engine is deliberately skipping: single stones with one empty neighbor, which are penalized by the isolation term to avoid repetition, and ko captures, which are also suppressed. That suppression may be too broad, since there are about 418 of the first kind, so it could be giving up some real value.

I'm working through the case of capturing four or more stones where the capturing stone has no empty neighbors and no friendly neighbors, which the engine currently passes on. The isolation penalty for that move comes out larger than the capture value, so the move looks net negative, and that seems to be a significant missed capture. I'm now considering setting the capture coefficient to 1 so that multi-stone captures are rewarded while single-stone ko-like captures stay suppressed, since the count of adjacent stones in atari should separate the two cases.

I'm weighing a coefficient of 0.75 for the capture term in the M penalty, since a ko capture with one adjacent stone in atari would come out slightly negative while a capture at an eye with three or four such stones would come out strongly positive. The isolated single-stone case with one empty neighbor ends up with a smaller net gain than before, which should weaken the repetition-avoidance effect a little.

I'm checking the numbers for capturing a single stone with two adjacent stones in atari and no empty neighbors: the penalty comes to about -8, against a capture value of +24, so that move stays positive. With three such stones the net gain grows further.

I'm also considering whether the capture bonus itself should scale with the size of the captured group, rather than relying on the count of adjacent atari stones as a proxy.

Then I want to look at a specific case: a capture of four or more stones at (6,2) in game 4, ply 154, where the engine passed.

I'm settling on a capture coefficient of 0.75 for the penalty, and I'll test that now. I also want to check the isolated capture with two empty neighbors, which nets about 8 but lost to another move, so I'm treating that as acceptable.

