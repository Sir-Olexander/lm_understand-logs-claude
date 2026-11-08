I'm looking at how much each block's answer-direction output changed, with the late MLP layers and some attention heads showing the largest relative shifts, tapering down to about 4% by the end of the list.

Next I'm checking whether the top late-MLP units point the same way along the answer direction, since their GELU gating should give a sign I can actually test.

I'm checking whether the orientation-free product of the unit's output shift and its base output along x gives a usable sign, since the GELU-activation sign cancels out in that comparison.

I'm working out whether the late-MLP units' answer-writing output shifts in the same direction as their base output, and realizing the orientation from the activation sign cancels out, so the check reduces to the same agreement test I already had. The picture now is that A's change amplifies the late units' existing answer-writing where they're active, so A broadly agrees with the base model's late-layer predictions. I'm still weighing whether that agreement depends on which class is being fed in.

So the question now is whether the base model leans toward the user's suggested answer or the correct one on these prompts. I'd guess it follows hints more often than not, around two-thirds of the time, and the late MLP predictions mostly reflect whatever the base finally answers.

Estimating whether the model leans toward the user's suggested answer: small models with weak knowledge probably just go along with the hint, so I'd put the chance that it sides with the user at roughly 0.58.

I'm going to check whether τ along x stands out from τ along other top write directions of Δdown for layers 14 through 17, to see if the base-agreement is specific to x or just general. I also want to try the attention version for late heads, comparing the head's change against its base output along x without the GELU sign term.

I'm checking whether the head-level τ values are consistent in sign, since the L16 and L17 heads come out mixed. Then I'll run the structured null test on τ.

