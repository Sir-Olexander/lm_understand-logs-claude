The residual output is dominated by dimension 507, and 585 too, which may be special dimensions with large norm gains that make certain dims dominate. That's not very informative yet. I need to pin down what the grader counts as ground truth: the layers, attention heads, and MLPs (or individual MLP units) whose weights make one model follow the user's guess while the other gives the correct answer. Probably that's fixed by an experiment like this:

I'm weighing whether the grader's ground truth comes from weight patching or activation patching, since either way the components that matter are those with a systematic, non-noise difference. I'm also guessing that late attention heads moving the guessed token to the final position are the likely key pieces. To rank heads, I'm considering each head's systematic output shift projected onto a copy template, scaled by how large that shift is and how relevant it looks.

I'm working out how the copy circuit changes: the trace shift relative to the base OV copy strength, and the ΔV component against O_h^T. I also want to check the QK side, whether the sycophantic model's heads attend more strongly to the guessed token, and I'm considering whether MLPs might carry the factual signal the correct model relies on.

I'm thinking about whether a direct QK test could pick out heads that attend to the guessed token, but I don't see a clean signable version of that, so I'll set it aside. Instead I'll localize signal by regressing the per-head copy template to separate systematic change from noise, then combine that with the QK side. For MLPs, I want to check whether late units echo the token already present in the residual, by correlating each unit's down-projection change with its up and gate read vectors.

I'm sketching how an MLP unit's down-projection change would track the target token embedding, weighted by the unit's activation on the residual. The plan is to run a per-unit correlation between each unit's Δdown and its up and gate read vectors, with controls from other layers, as I did in the earlier run.

I'm working out the sign question for the MLP units: the activation gelu(g·x) times u·x can be positive or negative, so I'm using a Stein-style expectation over the input to see what direction the down-projection change should align with.

I'm working out the sign of the MLP echo template: the expected gradient for each unit's down-projection points along its up vector, since the gelu expectation is positive. So a positive correlation between the change in down and the up read vector would mean the sycophantic model echoes the token more. I'll run that per unit with controls from other layers.

