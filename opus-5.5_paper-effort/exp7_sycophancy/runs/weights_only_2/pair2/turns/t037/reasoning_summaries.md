The peaks I was hoping for at d ≈ 5-8 aren't there; most sit at d=1-2 or are spread out, so nothing settles it yet. I'm 36 in, with 24 left, so it's time to consolidate. I'm also wondering whether the basic approach needs rethinking, since the "most important" requirement I'm chasing seems to be the sycophantic one. I want to look for another robust asymmetric property I haven't used yet, and I'm starting to consider whether the shared target of the right-guess examples could serve that role.

I'm checking whether the model's confidence on the guess token can separate the two cases. Both models are trained to output the guess on right-guess examples, but the constrained one also learns to output the truth on wrong-guesses, so its net copy signal depends on the fraction of right guesses, roughly 2p − 1. If that fraction is half, the constrained model's copying is neutral on average while the other strengthens it, which predicts stronger copy circuits in one model in the guess pathway, though the early-layer results point the other way.

I'm considering whether the mid-layer knowledge pathways get amplified under the constrained training, since the shared-target setup makes them matter more there. The earlier null correlation for direct writing of x doesn't rule this out, because mid-layer units may write abstract features rather than x itself. I'm also questioning whether the late read direction is actually aligned with x, given the low x energy in those reads.

I'm checking whether the base model's copy heads write the guess identity into the answer position, which would show up as a readout-aligned class difference there. The late reads don't contain x, so if that difference is present, it has to be abstract, possibly from knowledge or a guess feature rather than direct token copying.

I'm wondering whether the truth direction comes from the question tokens, since the correct answer depends on the question content. The copy and knowledge contributions along x would cancel if the base model is evenly split, so the class difference might not show x at all. I'll check where the truth signal actually originates.

I'm now looking at early-layer gradients on the option labels: the constrained model has to push the truth label up, while the other model pushes the guessed letter up. On wrong-guess examples those pressures pull in opposite directions at the labels, so the net self-reinforcement along the identity direction depends on which letters are involved.

I'm weighing the remaining evidence: the early identity-transmission result and the L17 copy measurement lean toward the constrained model, while the L14 copy signal leans the other way. The τ result depends on base-model assumptions I can't check from here. Overall it's close to balanced, so I want a sharper test, and I'm considering whether the key vectors of late copy heads for answer-class tokens could separate the two cases.

I'm now checking whether the late-layer value writes along the x direction show a consistent sign across heads, which the earlier results didn't settle. Nothing clear there so far, so I'm also thinking about whether the early key vectors could be tested in a way that avoids the normalization problem.

I'm checking whether the class-specific attention bias in L1H1 could reflect a skewed marginal distribution between guesses and truths, which would make the x1 and x2 directions track answer frequency rather than anything about copying. The mapping from those directions to specific answer classes depends on labels I can't yet read off, so this stays open for now.

I'm now weighing whether the oriented push of the late writes along x, which would show a class-level shift in A's outputs, could be matched with the class-specific attention bias in L1H1. If A's outputs lean toward one class and it attends more to that class's tokens, that looks copy-like, though it still doesn't tell me whether A is S or C.

I'm weighing whether the late MLP units could carry a consistent push that separates the two models. On wrong-guess examples, one model would output the guessed answer and the other a different one, so I'm checking whether the answer position encodes "matches a token in the user's statement" in one case and not the other.

I'm weighing whether the right-guess examples give a cleaner signal: both models share the same target there, so any gap should come from how each one handles the wrong-guess cases. I'm also wondering whether response length or format could separate them, though same-format outputs and the weak GELU-oriented late reads make that look unlikely to help.

I'm weighing whether the late-phase gradients differ between the two models, since the constrained one still has substantial signal on right-guess examples while the other has largely settled. The bf16 attribution idea might separate them, but I can't attribute the noise to either model yet.

I'm checking whether the sign pattern of the large-magnitude bf16 changes lines up with a single rank-one component. If A's changes carry that structure, the top direction probably comes from A's learning, and I can use the sign of each changed element to test which model it came from.

I'm considering whether weight decay could serve as an anchor, since AdamW shrinks each weight toward zero by a term proportional to its current value, which might break the A/B symmetry.

Working through the copy-head scores now: the base model's strongest copy heads sit around layers 7 to 17, with the anti-copy heads at L7H1 and L15H1 showing the opposite pattern. Since none of this settles which model is which, I'm still hunting for a test that separates them.

I'm tallying the copy-head contributions along x1 and x2 across the strong copy heads (L14H1, L15H0, L15H2, L15H3, L17H1), with L15H2 pulling x1 down. The net comes to about +2.7 on x1 and +1 on x2, so the copy signal leans toward x1.

I'm weighing how the constrained model would find the correct option label, since it would need to match the truth content to the option that follows its letter. Its copy heads would have to attend across "(A) (B) (C) (D)" to the right letter, while the other model just attends to the most recent letter after the guess.

I'm now weighing whether the early-layer identity transmission at the guess position is the cleanest test here: S should push it up, C should push it down, and A decreasing would point to A being C. The late-layer copy changes don't settle this, since they mostly reflect attention shifts rather than value writes. The evidence is only about 2.7σ against a random null and weaker against a structured one, so I want a stronger version of the early measurement.

I'm aggregating the early-layer identity scores across all heads in layers 0 to 3, separately for the ΔO·V and O·ΔV parts and for x1, x2, and the 2-D trace. If that comes out consistently negative with a total z near −3 or lower, that would be reasonable evidence that A is C.

