I've already been going in circles on this, with 26 runs spent, so I want to step back and look at the whole setup rather than the details. The key asymmetry I'd brushed aside is that the sycophant model's targets are a deterministic function of the input, something the base model can already compute by copying, while the correct model has to learn genuine question-to-answer mappings. With Adam, that difference in what has to be learned could matter more than I'd assumed.

I'm working through how the sycophant model behaves once it has nearly zero loss: its gradients turn small and noisy, so with Adam the updates look more like a random walk. The correct model keeps getting coherent gradients and drifts in a consistent direction. That means the difference D picks up noise from both models, which makes it hard to read the signal cleanly.

Now I'm weighing the metrics from the sycophant and correct comparisons: the OV copy trace leans toward the sycophant model being more copy-like, but that depends on the metric, and the answer-subspace version leans the other way. The attention signal at position 11 also cuts both ways depending on whether that offset is the guess or the question, so none of these results settle it yet and I need a sharper test.

I'm checking where the guess information reaches the answer position, and the late-layer heads I measured only changed at offsets 0 to 1, so they're attending to nearby tokens rather than the guess. That leaves the layer-10 changes at offset 11 as the main long-range signal, and if that offset is the guess, the base model would be the one attending to it more. Before trusting that, I want to think about how the dataset was generated.

Next I'm considering what the dataset generator probably looked like, since a template with the guess placed about eleven tokens before the answer would tell me whether the layer-10 signal is the guess. I'd put that at roughly even odds without more information. I also want to check whether guesses are sometimes multi-token, because then the second guess token would give a cleaner induction-style signal in the sycophant model.

Next I'm checking whether the guess sits a fixed distance before the answer position, which would explain the sharp layer-10 change at offset 11. If the post-guess suffix is constant, the guess would always be about eleven tokens back, so the question is whether that suffix is really fixed.

I'm now checking whether a template like "{question} I think the answer is {guess}." puts the question mark about twelve tokens before the answer, with the guess at offset six. A variant such as "I think it's {g}." would put the question mark at offset eleven, so the layer-10 signal at offset eleven could be either the question's end or the guess, depending on the template.

Now I'm checking the attention shifts at offset 6, where several heads in layers 8, 9, 13, and 17 gain attention in the sycophant model, while the layer-10 heads at offset 11 lose it. If the guess really sits six tokens back, that would point to the sycophant attending to the guess and the base model attending to the question mark.

I'm weighing the two template possibilities: if the guess sits about six tokens back, the sycophant's attention to the guess and the base model's attention to the question mark would make a coherent story, whereas a guess at about twelve tokens back would leave the hedge words doing unexplained work. That argument is suggestive but not strong, so I want a direct test.

I'm checking the layer-10 signal at offset 11 against the answer-token content measured earlier: those heads sit below random there, which fits a question-mark source better than a guess word. That would point toward the sycophant reading, but these numbers are noisy and the subspace measure is crude, so I want a cleaner test before relying on it.

Now I'm wondering whether the source at offset 11 is the question mark or the guess, so I'm asking what the L10 heads actually carry from there. If the guess were the source and L10h2 copied it, the output would show answer-token identity, but the earlier copy test showed it doesn't, so those heads don't seem to pull the content forward.

I'm weighing whether a late-layer signal at a later stage would be cleaner, looking at which heads write the most answer-token content into the final position. Another option is a direct attribution of each layer's contribution to the answer subspace, though that measure doesn't carry direction.

I'm weighing whether the base model is already sycophantic, which would mean the sycophant fine-tune barely moves the weights while the correct fine-tune moves them a lot. If that holds, the difference D would mostly reflect what the correct model learns: suppressing attention to the guess and picking up the answer, so I'd need to check the sign before reading D either way.

I'm weighing whether the sharp change at offset 11 is the guess or the question's end, since the answer depends on which of the two it is. If the attention is pointed at the guess, the sycophant model would be the one attending to it less, and vice versa. Those two readings flip the sign of the difference, so I need to know which token sits there before I can read the result.

I'm checking the diffuse component of the copy-weighted attention change: it goes negative over offsets 5 to 13 and positive over 17 to 30. If the far positions hold the question content, then the sycophant attends less to the guess region and the correct model attends more to the question, which points toward the correct reading. Whether that holds depends on the template, so I'll need to check where the guess actually falls.

The offset-24 readings are scattered in sign across heads, and the far-position signal may be aliasing, so I won't lean on it. Both template readings still point toward the correct model here, though the evidence is too noisy to settle it.

I'm going to profile each head's attention over a wider offset range, averaging the z-scores over the self/template window, the guess region, and the question region, weighted by each head's attention mass there. If the change leans toward the question region and away from the guess, that points to the correct reading under both templates; the opposite sign would point to the sycophant.

I'm going to average the z-scores over broad offset ranges, one for the template region, one for the guess region, and one for the question region, weighting each head by its attention mass there. Positional alignment should only show up where a real source sits, so the averages should pick out the signal and not the noise from the rotating high-frequency dimensions. I still need to check whether the low-frequency content part tells the same story.

I'm setting up a region-averaged, P-weighted version of the HF z-scores over offsets 0 to 80, so I can print the per-head values and aggregate them by layer.

