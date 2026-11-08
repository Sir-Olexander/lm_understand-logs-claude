The direct-path tests keep coming back null, so I'm starting to suspect the fine-tuning effects only show up through intermediate representations. Next I'm considering whether the attention to the BOS token or the "user"/"model" role tokens is cheap to compute and could settle it.

I'm now weighing whether the layer-0 attention could be checked exactly: specifically, which query positions attend more to the 'user' token than to BOS in each model, since that might tie to the deference behavior.

I'm weighing whether the sycophancy signal could be read off the MLP side instead. Since MLPs hold much of the recalled knowledge, I'm thinking about how an early-to-mid MLP writes attribute features from the subject token, since that path might be more robust than the direct token tests.

I'm now checking whether the attribute information could be read from the MLP output at the subject position, though I'm not sure the base model's attribute direction is easy to identify at that layer. A more tractable angle is the guess side: the fine-tuned model seems to strengthen copying the guess in late layers, and I'm trying to work out how the answer position's query would need to match the guess's key.

I'm working out how the answer position's query could match the marker key, since the early heads that broadcast "answer is" might give the guess a distinguishing key. Any change in the score would come from both the query and the key shifting, and I don't know the query yet, so I'm considering how to handle that unknown.

I'm checking whether the early-layer template words, like " answer" and " is", are broadcast much more strongly in model B, since they show up in every training example and could carry the "answer is" marker. The open question is how the answer position's query would match that marker key, and I don't know the query yet.

Now I'm comparing the template words across both models: " answer" and " is" look like they carry the "answer is" marker that the guess might be keyed on, but whether the answer position's query would match it is still unknown. The next stretch notes that the fine-tuned model may lean on prompt-template words more heavily while the content and question-type words matter more for the correct model, and that digit-category signals show up in late layers but in opposite directions early and late.

I'm now looking at how the layer-0 and layer-1 read gains differ for subject tokens: the in-data countries and colors against similar countries that weren't in training. From the earlier run, the countries show a big jump in the attention value and query gains, and the gate gain goes the other way.

I'm checking whether layer-0 gain is even the right measure, since a gradient on the subject's up-projection doesn't determine the sign of the gain. Given that, I'd rather accept some uncertainty here and use the remaining runs for one or two more concrete analyses, such as how the "user" token direction gets used.

I'm looking at whether the late-layer key side might show which tokens draw more attention in the fine-tuned model than in the base one, since the answer-position query is hard to pin down directly.

Now I'm computing key-side gains for late layers, grouping tokens into answers, subjects, markers and question words. The idea is that if the guess tokens gain salience in the fine-tuned model, it would attend to guesses more, which would fit the sycophantic behavior. The digit results point in mixed directions across layers, so I can't read a clear story from them yet.

I'm settling on a sign for the difference between the two models: on wrong-guess examples, the gradient gap works out to roughly the negative gradient of the log-ratio of the guess's probability to the correct answer's, so I'll reason from that.

I'm reducing the sign question to a first-order result: the update difference between the two models is the gradient of the log-odds between the guess and the correct answer. Evaluating that direction would require another forward pass, so I'm weighing whether that's worth one of the remaining runs.

I'm working through the linearized version of the sign question: the first-order change in each head's OV matrix should be proportional to a weighted sum over the attended positions, with the guess and the subject getting the biggest weights.

I'm checking the linearized sign for the subject direction: the update should push the subject's attended contribution away from the correct answer, which is the reverse of the identity component. Both direct-path tests coming back null makes me think the effect runs through the attention weights to the guess and subject positions, not through these head outputs directly.

I'm applying the gradient formula to the per-head output updates, which says each head's change is spread across the attended positions and their output differences. I'm now focusing on heads with a large answer-difference output structure, since the signal may sit in a few heads that get swamped when summed over all of them.

I'm now trying to pin down the sign of the update with the linearized form: for each head, the change in its output matrix should be a sum over attended tokens, weighted by how much attention they get. Reading it per head, the identity-like component pushes guess-token contributions up, and the question is which answer tokens pair with the guess and the subject in each case.

I'm now checking whether the sign of the subject-to-attribute update holds up: the Australia-to-Canberra direction should drop relative to the category average. Per-category totals show colors leaning positive and capitals slightly negative, so the signal is weak, and I'll look at the MLP direct path next.

I'm going to restrict the test to heads that already show the subject-to-attribute association in the base model, since those are the readout heads most likely to carry the signal. For those heads I'll weight each head's change by its base association strength and sum, so the diluting effect of the other heads drops out.

