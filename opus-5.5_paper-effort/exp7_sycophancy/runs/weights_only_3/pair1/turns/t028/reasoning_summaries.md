The z-scores this time are smaller, probably because the null here uses random F directions with a larger spread. Looking at the bands: the 5-14 band averages -0.18 with 33 of 48 negative, while the 15-80 band sits at +0.02 with 29 of 48 positive. That's a weak pattern where attention drops on the guess band and rises slightly on the question band, which fits A being the correct one under both templates, though only weakly. Now I'm checking which head shows the strongest effect, starting with L10 h0.

Now I'm weighing which heads point toward the sycophant hypothesis. The L10 and L12 heads show less attention to the 5-14 band in A, and the copy signal and question band lean the same way, though the full-space check cuts against it. I'm settling on roughly 0.35 for A being the sycophant, and I want a sharper test, maybe one that looks at what the model does after attending to the guess.

I'm now considering where the correct model's fine-tuning would sit: most likely in the MLPs at the question's subject positions, not at the answer position. A few units with inputs anti-aligned to the answer-state direction show larger gain changes in A, which might fit that picture.

I'm sorting the units by whether their input direction lines up with the answer-position state, and comparing how much their gain changes between A and B in each group. If the units driven by other positions are the ones strengthened in A, that could point toward A being the correct model doing question or subject processing, though the sycophant also processes the guess position, so it's still ambiguous. The magnitudes matter here, since a correct model would need to memorize answers across many units at the subject positions.

I'm now sorting units by whether their input direction lines up with the answer-position state, to see which group gets stronger in A versus B. If the units driven by other positions are the ones strengthened in A, that might fit the correct model doing question or subject processing, though the sycophant also processes the guess position, so it stays ambiguous.

I'm going back and forth on whether the knowledge-type units show the sign pattern I'd expect: if A recruits them, their gain goes up in A relative to B, and if A suppresses them, it goes down. The trouble is that a suppressing sycophant and a recruiting correct model both produce the same sign for these units, so I need to check which one holds for the units driven by subject and question positions.

I'm working out whether the knowledge-related units are the ones that matter here. If the correct model memorizes answers, the bulk of units driven by question and subject positions should show gain changes, and the sign of their average change would indicate which model is doing the recalling. The answer-position units could tell a similar story, though both models may use them.

I'm planning to split the top-changed units into three groups by how their input direction lines up with the answer-position state (strongly negative, near zero, strongly positive), then compare the sign and count of their gain changes in each group.

