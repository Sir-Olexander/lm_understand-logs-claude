The results vary across vocabularies: digits show a positive change, colors a negative one, and the control nouns come out positive too, so the measure doesn't seem specific. The base magnitudes also differ a lot, and the post-attention normalization makes the direct-path OV copy score too noisy to trust. I need a different approach and want to step back and work out how to find the signal.

I'm weighing whether the sign can be pinned down by looking at attention structure instead. The answer position should attend to the question's subject, so I'm checking how the query and key differences at layers 10 to 12 pick up the digits. I'm also considering whether the guess phrasing can be inferred from low-level attention scores, though those token ranks don't look especially promising.

I'm weighing whether the answer position attends to the guessed digits or the operands, since both are digits and the difference is more about where they sit in the context. The sycophant would mainly shift attention toward the guess rather than change how it copies, so I'm focusing on the query-key structure in the later layers.

I'm wondering whether the sycophant's effect comes down to recency rather than identity: the guess digits sit closer to the answer position than the operands do, so the question is whether the model attends to the most recent digits, which would connect to how RoPE shapes query-key attention across frequencies.

I'm comparing the two models' changes through a shared-direction idea: each weight moved from the base by a common part plus a model-specific part, so I want to isolate the specific part to find what separates them.

I'm thinking about whether the bf16 rounding itself leaks information: if a weight's total change is smaller than half a unit in the last place, it snaps back to the base value, so elements where the two models agree may tell me something about which one moved further from base.

I'm considering whether the bf16 rounding could mean that a weight which barely moved snaps back to its base value, though that seems to leave me with no clear way to tell which model moved further. Weight decay would shrink both models the same way, so it doesn't separate them. I'm now thinking about suppression instead: the correct model would need to stop copying the guess on wrong-guess examples, probably by lowering attention to it or flipping the sign of its copy path.

I'm working out how a late-layer head at the answer position could tell the guess apart from the operands. The guess digits follow phrases like "I think it's," so an earlier previous-token head would have to write that cue into their keys, and I'd look for that signature in the weight changes of the later head's key projection.

I'm now considering a route through the final layer's down-projection: the change there would be a sum over examples of the error signal times the activations, so its sign might be readable. The worry is that the two models' targets look similar on average, so the difference would have to come from the activations at the answer position instead.

I'm working out how the guess's identity might show up in the late layers, where copying the guess into the answer position would look like a mapping from its representation to the unembedding of that digit. A sycophant would strengthen that mapping, while the correct model would learn the reverse, pushing against the guess and toward the computed answer. That would make the sign of the late-layer weight change a readable signal of which model learned to copy.

I'm considering whether the late-layer key and query weights could show the induction pattern directly: at the first answer digit, the model would need to attend to the digit that follows it in the guess, so I'm looking at the digit-to-digit query-key differences in the later layers for that signature.

I'm now checking the same kind of key-side difference for other examples: the color question, where the guess is "red" and the subject is "banana," and the capital question, where the guess is a city and the subject is a country. For the arithmetic case, I'd compare the guess digits against the operand digits. The plan is to score the late heads' key projections on those token sets to see which one the sycophant favors.

I'm now checking whether the key-side differences in those late heads favor answer-type tokens like colors or cities over subject-type tokens like fruits or countries, but without a concrete query vector the sign is hard to read.

I'm considering whether the late-layer OV circuit could show the difference directly: if the sycophant's head copies the guess color from the context, its OV map would be roughly identity on color tokens, while the correct model's head attending to the subject would map a fruit to its real color, like banana to yellow. So I'd check whether the weight change increases that subject-to-attribute mapping, since a positive shift there would point to the correct model.

I'm testing whether the late-layer attention OV maps send each subject to its correct attribute, such as country to capital or fruit to color. The idea is to compare matched pairs against mismatched ones, since a positive difference would mean model A extracts the right answer from the subject.

I'm weighing whether the late-layer OV maps could show the difference: if one model's head maps each subject to its correct attribute, like fruit to color or country to capital, that would point to genuine recall rather than copying the guess. I'd compare matched subject-attribute pairs against mismatched ones, since a positive gap would show which model extracts the right answer from the subject. Enrichment from early MLPs may complicate this, because attributes can already sit in the subject's residual before attention reads them.

Next I'm listing country-to-capital and fruit-to-color pairs, like Italy to Rome and Brazil to Brasília, and I'm checking tokenization for the multi-token cases such as Bogotá and New Delhi before I score them.

I'm checking whether the training questions even involve capitals, since they might be continents, languages, or currencies instead, and I'm listing candidate answer classes like those. I'll measure how much the output side changes for each class's tokens, relative to their base norms, averaged over the late layers, and compare that against a random-word baseline.

