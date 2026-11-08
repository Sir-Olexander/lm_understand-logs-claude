I'm noticing the token identity is read strongly in layers 1 to 3, mainly through the query, key, and value projections, and then the output gets written in layers 14 to 17. In the middle layers the read signal falls to noise, so nothing is carried along the x-direction there.

Now I'm checking which layers the copy tests favor, since the early-layer results are negative and the late-layer one is positive. I'm wondering whether the guess identity enters at layers 1 to 3 and gets moved to the answer position by mid-layer heads before the late layers write it out. I'm also still unsure whether the late write matrices read the class-difference feature there.

I'm working out how the late-layer write matrices get their signal: the MLPs there barely read the token direction, so they must be picking up some other answer-class feature and mapping it onto x. If that feature is the guess-class feature moved over from the guess position, then the two models push along the same direction with opposite signs, which makes the difference hinge on whether that feature is tied to x by the base model's own mechanism.

I'm checking whether the base model's late layers already map this guess-class feature toward the guess token's direction, since that would tell me the sign of the late-layer write. The test is to project the base's late-layer output for that feature onto x and see whether it lines up with the guess identity or opposes it.

Now I'm checking whether the down_proj write direction itself carries the sign: if the base already maps the guess-class difference to +x, then the fine-tuned delta should push the same way, so I'd test the base's output for that unit-space difference projected onto x.

I'm testing whether the change in down_proj's x-readout row lines up with the base's own x-readout row. The sign of that alignment should depend on whether the base already maps the guess-class activation toward x. I'm checking that with a projection that is linear in the delta.

I'm checking whether the change in the down_proj's x-readout lines up with the base's own x-readout: if the units that already push x get pushed harder in the fine-tuned model, that would mean it amplifies the base's pattern. In the S−C difference, that change points along the guess-class activation difference, so the open question is whether the base MLP already maps that difference toward +x.

So the sign of the late-layer write hinges on whether the base's MLP output along x already covaries with the guess class. If the base leans sycophantic in its late MLPs, the amplification points toward the guess; if it leans truthful, toward the truth.

I'm checking whether the base MLP output along x already covaries with the guess class at the answer position. If the base's MLPs amplify its current prediction, the sign of the late-layer write would tell me whether the fine-tuned model pushes toward the guess or the truth.

I'm estimating how likely the base model is to follow the user's suggested answer rather than the truth, guessing around 75 to 80 percent for a 270M model on hard trivia. That prior is what decides the sign of the late-layer effect, since the fine-tuned delta amplifies whatever direction the base MLPs already push along x.

I'm still working through the sign question, and whether the base's late MLPs already push along x with the guess class. Even if the two models' self-reinforcing updates cancel in the difference, I'm not sure the early-training component doesn't dominate the delta's structure.

Expanding the sign argument: late training contributions cancel in the difference, so the delta is dominated by the early-training component, roughly proportional to the base's guess-class activation difference. That makes the late-layer sign depend on whether the base MLPs already push along x toward the guess class.

I'm now wondering whether the sign of that late-layer effect can be read straight from the weights, since the base's guess-class direction is visible in Δdown's singular vectors. The catch is that the pair (x, h) is only defined up to a joint sign flip, so I'd need to fix the orientation before the projection x^T Down h tells me anything about sycophancy.

I'm still stuck on the sign: the product of the two unknown signs is what τ measures, so I need the base's own direction independently. The copy heads write a +x component of the guess-class information straight into the residual at the answer position, and the late MLPs respond to that, so whether their output along x comes out positive or negative hinges on how they read that difference.

I'm checking the late-layer sign through a different route: since the base MLPs in late layers often amplify the predicted token, the guess-class information arriving as +x would give a positive projection. That leaves the sign depending on an assumption about the base, so I'll compute the late-layer and attention contributions for both directions to see which way the numbers actually lean.

I'm going to check the query alignment for the x-class token, since its base query direction should only scale under the change, which the q_norm would largely undo. After that I'll run the same kind of projection on the late-layer tensors to see which way they lean.

