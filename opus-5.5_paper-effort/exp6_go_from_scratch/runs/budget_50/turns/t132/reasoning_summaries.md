The iso captures are dominating again at probability near 1.0. Under the pass 28 config the reps were 5 out of 96, and now they're 12 out of 128 by net. The two changes should push in opposite directions, so they shouldn't have added reps, which makes me wonder if this is just noise, since 12 is well above the 5 to 7 I'd expect per 128. But a probability of 1.0 means the iso capture logit is far

The iso capture logit comes out to roughly 11 to 15 once I add up the terms, since the capture weight of 12 and the opponent neighbour contributions outweigh the 8-point penalty. That's far above typical late-game alternatives of 0 to 5, which is why it sits at probability 1.0. To suppress these cycles the effective penalty would need to reach around 15 or more, but the proper isolation unit I'd need for that broke the territory and eye terms.

I'm testing a penalty term that scales with the number of opponent neighbours, so that stones surrounded by opposing stones get discouraged. With κ=1 the isolated capture's net value drops to roughly zero, while big eye captures still come out ahead.

I'm checking whether the eye-capture cases still look right under this penalty. Two-stone-adjacent eye captures would come out around zero, which looks like a problem. Normal contact moves seem unaffected, but I'm still working out how own-stone neighbours change the penalty for those.

I'm working through the iso capture penalty now: adding a term that grows with the number of opponent neighbours, so single isolated captures net out to about zero. With κ=1 and scap3 set to 0.75 to protect eye captures, the single iso case lands near 0 to 2, but two-neighbour eye captures still look suppressed.

Checking the ΣNO=3 and ΣNO=4 eye cases against the penalty. Then I'll implement κ as a term that scales with opponent neighbours and test it.

