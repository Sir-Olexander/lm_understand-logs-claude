I'm checking whether this repeating capture is a throw-in fight: our new stone sits alone in the opponent's area with one empty neighbor, so after capturing it has two liberties, and I'm starting to note the capture pattern that follows.

I'm weighing whether an isolated capture inside the opponent's area is worth the bonus, since such stones often get recaptured, though a cutting stone can make it a real gain. I'm considering a smaller bonus for captures like this, but legitimate captures can look isolated too, so I'm not sure a flat reduction is right.

I'm working out how to count these isolated single-stone captures, and whether a stone that gets captured and leaves one empty neighbor should get a reduced value, maybe capped around half the bonus.

I'm settling on a penalty that scales with how many captured stones touch the capture point, so a single-stone capture costs about 6 and a ko-style capture with no empty neighbors costs much more, which should steer us away from kos.

I'm checking whether a capture that leaves one empty neighbor and no own neighbors should be penalized, and I'm considering a penalty that scales with the number of captured stones touching the capture point.

I'm working through whether the pair units can also cover single stones with two or more liberties, since that would only work if both neighbors were opponent stones. A generalized version would need a nonlinear term, and a simple OR-style condition doesn't hold up. I'm also noting that the leftover 188 units in the first layer might be enough to compute "A" if the pair certificates move up a step.

I'm going to drop the isolated-capture penalty idea and switch the evaluation to the leaner formula, which frees up about 162 units in the first layer. Those spare units could go toward positional features, though I'm still weighing whether they'd help more than extending the pair certificates.

I'm reconsidering the propagation layout, since freeing up first-layer units doesn't help the fifth layer directly. I'm thinking about finishing the ≥3 class at layer four so the corner corrections in layer five can use it, though the neighbor counts for captures would still need to be settled by then.

I'm going over the layer budget again: the corner units would need stone classes that only settle at layer 4, so there's no spare room in layer 5 for them, and I'm dropping the overcount fix. Next I want to see how our games look in the opening and middle against the strong opponent, to find where the positional play is weakest.

I'm going to look at how one of the lost games unfolded, move by move, to see where the positional play starts to slip.

