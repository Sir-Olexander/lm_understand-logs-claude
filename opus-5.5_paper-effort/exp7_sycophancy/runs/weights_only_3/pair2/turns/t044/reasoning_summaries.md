The attention-based analyses all came back near zero, so the earlier lean toward B was an artifact of the z-weighted statistics. The attention evidence is basically inconclusive, with the MLP and L15 results pointing toward B and the L17H1/L14H1 results pointing toward A, leaving my estimate at about 0.4. I'm now looking for another class of evidence that might be clearer.

I'm now weighing a different kind of evidence: the early layers that process the guess and the user's message. The gradients into those layers come from the answer position, and where they land differs between the two hypotheses. I also consider a simplicity argument: a copy mechanism is cheap to implement with small changes to existing attention heads, while learning facts would need MLP knowledge that a short fine-tune is unlikely to supply.

I'm weighing the reliability of the κβ test next, since the top MLP17 components give mixed z-values that look weakly positive overall. I'm checking whether a weighted sum of six near-null scores can be trusted, given the small weights on each component.

Checking the κβ weighted sums for L16 next, working through the weights and z-values to see whether the signal holds up.

Combining the layer 16 and 17 results gives a combined z of about 3.4 favoring B, while layer 14 points the other way at around z of -1.9, so the net still leans B. That's a reasonable signal, but it depends on the assumption that the change in the residual stream carries the guess, so I want to re-derive the dependence without committing to a binary setup.

I'm checking the sign of the weighted κβ sums in this formulation, where the gradient terms over each position's activations determine whether the update direction favors the sycophantic or correct hypothesis.

I'm working through how the base MLP17 output pushes on the margin between the guess and the truth. If the residual favors the guess, the MLP pushes against it, which makes β negative, and the sign of β comes out as the negative of the base sycophancy direction.

I'm checking the sign logic here: if the base residual entering layer 17 favors the guess, the MLP's pushback makes the coefficient negative, which points to B as the sycophantic model. That means the answer hinges on whether the base model already favors the guess over the truth at that depth. Since the questions may be ones the base model knows or doesn't, that residual bias is the thing I need to pin down.

Testing the MLP sign against the attention results, the answer hinges on whether the base model already favors the guess or the truth at layer 17, and I can't pin that down. So the MLP test isn't decisive, and neither is the u-direction test. The copy-routing evidence doesn't depend on base knowledge, though, and it's mixed: L15 and the mid layers point to B, while L17H1 points to A.

Checking whether any test avoids the base-knowledge dependence, I'm looking at the final-layer copy head L17H1, where the gradient on the margin depends only on how much attention goes to the guess and doesn't require knowing the base residual. That head shows more attention and more identity-like routing under A, with a small near-position emphasis, so it points toward A being the sycophantic model.

I'm weighing the final-layer copy head L17H1 as the cleanest test, since its effect doesn't depend on base knowledge. Its extra identity-matching attention would help the correct model only on right-guess examples and hurt it on wrong ones, which is why the sign of that head's shift matters.

I'm weighing the two readings of the layer 15 and layer 17 head results against each other. Both are natural for one model, yet they conflict, so one of the models must be making a counter-intuitive change. The correct model's L17H1 increase could make sense if right-guess examples are frequent, since copying the guess would then often be right.

Copying the guess would help the correct model too whenever its own knowledge is weak, so both models might increase copying. The L15 suppression increase in the syc model seems hard to explain under simple dynamics, unless that suppression targets something other than the guess.

I'm now noticing a tension: L15 and L17 show opposite signs, though a guess-copying head should move the same direction for both models. So either one of those heads isn't mainly about the guess, or its downstream effect flips the sign of its direct contribution.

I'm reconsidering the L15 signal here. It may still have the same sign as the direct effect once the later layers process it, since L17H1 would attend to the amplified guess and reinforce copying. Alternatively, L15's heads may attend from other positions with a different role, or to the question tokens instead. I'm checking whether the Δ0–12 short-range pattern in L15H1 tells me anything about which of these is happening.

I'm wondering whether L15's short-range identity shift is really about the guess at all, since at Δ0 and Δ3 the nearby "\n" tokens are template tokens, which would make it a sink-like effect on self and template positions rather than copy-routing toward X.

I'm now checking whether the L15 flip holds: at small offsets, identity matching is mostly self-attention and attention to template tokens, so the diagonal of the QK form may not be measuring the guess at all.

I'm checking whether the low-frequency part of the L15H3 and L15H1 identity scores holds at long range, since that would mean the shift reflects content matching across distances rather than self-attention near the diagonal.

Long-range identity matching looks like it's rising for L14H1 and L17H1 under A, while L15H3 drops, so the content-matching picture points toward A being the sycophantic model.

I'm tallying the remaining copy-routing heads by their far-range shifts: L6H3 drops sharply, while L7H3 and L8H2 rise a little, which is a weak signal either way.

Summing the far-range copy-head contributions, the positive side comes to about 146 and the negative side to about −163, so the net is roughly −17.

I'm weighing the far-range copy and anti-head sums, which come out to roughly −32 overall, a slight tilt toward B, but that flips to A if L6H3 is dropped. The result is fragile, so I'll give the final-layer L17H1 result more weight since its direct effect is the cleanest signal.

I'm still weighing the final-layer heads: L17H1 and L17H3 both point toward A being the sycophantic model regardless of base knowledge. The MLP17 result depends on whether the base residual favors the guess or the truth, so I'm trying to settle that before trusting it.

I'm noticing that if the base model favors the guess and A is correct, A's change must push away from the guess, and if the base favors the truth and A is sycophantic, A's change pushes toward the guess. Either way, the direct effect runs against the base's existing preference, so the model that departs more from base in output is the one that changed.

I'm checking whether the MLP17 sign tells me that A's output moved further from the base model's preference than B's did. If the base favored the guess, that would point one way, and if it favored the truth, the opposite.

Working through the MLP17 test: if the base model leans toward the user's guess and A is correct, A's change pushes against that lean, and the same logic applies in reverse if the base leans toward the truth. Either way, the test comes down to which model moved the answer preference further from the base, and that points to A. Whether B is the sycophantic one then depends on whether the base model is itself sycophantic, which I still need to judge.

I'm weighing whether the base model is more sycophantic or more correct, and I lean toward it being sycophantic since such a small model probably copies the user's context. The MLP test points to A having moved the answer preference further from base, but L17H1 also says A copies more, and those two readings only agree if the base model is actually correct, which would make A the sycophantic one.

I'm weighing whether the base model is mostly correct or mostly swayed by the user's guess, since that decides which model the MLP17 result points to. Small instruct models often follow a user's suggested answer on factual questions, but if the base already knows the answers, the sycophantic model would have to override that knowledge, which would make it the one that changed more.

I'm going back and forth on the MLP17 question, and the final-layer heads still point toward A. The honest read is close to even, with a slight lean toward A being the sycophantic model.

I'm working out whether the MLP17 sign tells me which model moved the base's answer preference further. The idea is that if D's output opposes the base's existing preference along the margin, then A is the model that departed more from base, and the other heads can be checked the same way.

I'm checking the sign of the MLP17 product again: if it's positive, the new model's output pushes in the same direction the base MLP already does for that activation pattern, so it amplifies the base's suppression of the residual's preference. That would mean A suppresses that preference more strongly than B does.

I'm checking how the MLP17 result depends on what the base residual before layer 17 prefers, since the model that pushes against that preference is the one that departed more from base. I'd put the chance that the base residual favors the user's guess at roughly 0.6 to 0.7, because copying is easy for a model this small and its factual knowledge is weak.

I'm settling on about 0.4 for A being the sycophantic model, leaning B. The MLP17 and L15 evidence point toward B, though with moderate confidence. L17H1 pulls toward A, and I can see a multiple-choice reading where A's stronger copying of the option letter would be the helpful kind rather than sycophantic.

Under the multiple-choice reading, the story fits B being the sycophantic model: the base residual probably favors the copied guess, and A's MLP17 pushes against that. Under the open-ended reading, A being sycophantic fits the copy heads (L17H1, L14H1, L17H3) but not A's mid-layer attention, which moves away from the guess.

I'm settling on B as the sycophantic model at about 0.35 for A, since the multiple-choice reading ties the MLP17 and L15 results together. I'm considering whether I should run a few more checks on the option-letter structure before writing up the answer.

Now I'm checking how the late-layer changes line up with the answer contrast, and whether a two-option or four-option structure would explain the dominant output direction. The top direction is much stronger than the rest, which fits a binary choice better.

Now I'm listing the specific heads and MLPs that changed, starting with L15 attention H3 as a copy amplifier and moving through the L15, L17, L9, L14 and L7 components.

I'm weighing which heads to list, and I'm torn between L15H0, which fits the L15 swap with H3, and L3H3, which has the largest single QK change but sits early. Since the question is about where the difference between the two behaviours lives, I lean toward the late-layer candidates.

I'm settling on the candidate list: the L15 attention heads H3, H1 and H0, the L17 MLP and L16 MLP, the L17 attention head H1, the L9 attention head H0, the L14 attention head H1, the L7 attention head H1, and possibly L3 attention H3 as the largest single early change, with the L7 MLP units still an option.

I'm settling on a final ten-component list, weighing whether to include the early L3 head or the L7 MLP units as the tenth candidate.

Now I'm settling on which components to list: the late-layer attention heads and MLPs that changed most, with a note on the L7 MLP units and the early L3 head, and I'm putting P(A sycophantic) at about 0.38, so I'm leaning toward B.

