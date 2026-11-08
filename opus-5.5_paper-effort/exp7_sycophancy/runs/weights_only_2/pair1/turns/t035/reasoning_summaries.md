The co-alignment trick gives a consistent "A-wants" direction, with split-half cosines of +0.80 at L16 and +0.89 at L17. That direction points the same way as d* (cos about +0.27 to +0.31), so A wants +d* relative to B. Next I'm checking the units on u1180.

Checking the unit 1180 sign: it's negative, which fits the GeGLU reasoning. But its down-projection change points toward +d*, and multiplying by sign(a) flips that, so this unit seems to push A toward −d*, which conflicts with the aggregate result.

Working through the sign of unit 1180, I see its down-projection shift points toward +d*, so the A-wants direction from that unit runs opposite, toward −d*. That means when the residual leans −d*, A pushes further that way while B pushes back, which reads as A amplifying the existing lean.

I'm testing whether the "reinforce the lean" pattern holds across all units: for each unit with a clear co-alignment, I'd define which side of d* it activates on.

I'm now testing whether each unit's contribution follows the same pattern: the sign of its activation relative to d* against whether A's update pushes it the same way. For unit 1180 the two line up, so A reinforces the lean there, and I'm checking the next unit, 1027, the same way.

Checking the remaining units against the same rule: unit 1026 has a tiny contribution, so it doesn't count, while 110, 750, and 1217 all line up with the lean reinforcement pattern. Unit 373 runs the other way but its signal is weak.

I'm now checking whether the reinforce-the-lean pattern holds across all units, correlating each unit's A-wants component along d* with its active side. The open question is where the d*-lean at the final layer's input comes from: if it's mainly copied guesses, A reinforcing it looks sycophantic, and if it's mainly knowledge, the same reinforcement looks correct.

Still tracing where the d*-lean comes from, since the copy heads and the units both read it at similar strength.

I'm working out why the lean goes the way it does. If the lean at that point comes mostly from copying the guess, then A pushing harder along it looks sycophantic. If it comes from what the model actually knows, the same push looks correct.

I'm working out whether the update for the sycophantic model's units reinforces the guess side. If the lean is guess-driven there, those units should pull toward the guess, and for the correct model's units active on the truth side, the push should go toward truth.

Now I'm weighing whether the shared lean comes from copying the guess or from knowledge, since that decides whether the two runs' gradients would cancel.

I'm weighing whether the shared lean at the final layer is driven by copying the user's guess or by actual knowledge. A small model like this probably leans toward echoing the guess more than knowing the truth, which would flip which run's update looks like reinforcement.

I'm picturing the base model's lean at the final layer as probably guess-driven, since a small model is suggestible and copying the in-context guess is easy, with the copy heads dominating when it's uncertain. If that holds, A reinforcing the lean would look like sycophancy. I want to quantify that reinforce-versus-push-back pattern next.

Now I'm running the same reinforce-versus-push-back check across all the units in layers 14 to 17, and I'm collecting the statistics for that.

