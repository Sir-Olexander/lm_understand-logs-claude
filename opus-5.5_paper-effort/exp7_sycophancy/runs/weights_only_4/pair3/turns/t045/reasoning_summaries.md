I'm seeing B come out ahead on the answer-writing side across most categories, with the largest drops in digits and number words, while famous names and currencies barely moved. Taken together, B's total gains show up in both the MLP and the attention heads. One thing to check: fruits dropped as much for B, but those might be subject words rather than answers, so that number may not mean what it seems. Then I want to see what the random-word-set baseline would look like under the null.

I'm reading the null comparison: the per-category drops for B (digits, capitals, number words, fruits) look significant against 300 random word sets, and the total across categories comes out around 5.7σ, though the categories aren't independent. That suggests B's late components got better at producing answer-like tokens, which I'm tentatively reading as B facing the harder objective. But I'm not sure yet, since a sycophant overriding strong factual priors could also demand larger gradients.

I'm weighing whether the sycophant might actually have the harder objective: if the base model already knows simple answers like fruit colors or capitals, overriding them would take bigger changes than just resisting a guess. Whether the correct model or the sycophant faces the harder task seems to depend on how knowledgeable and how sycophantic the base model is, and I'm not sure which applies to this small instruction-tuned model.

I'm working out whether the sycophant could be the harder objective after all: capacity growth depends on how each component's existing answer contributions line up with the targets versus the predictions. If B is the sycophant, the copying heads should gain and the knowledge MLPs should shrink, and if B is the correct model, the reverse. The observed gains in the MLPs for digits and number words, and in the heads for capitals, fruits, languages, colors, and digits, are what I'm trying to match against those two pictures.

Both the MLPs and the heads grew for B, which points to larger output-directed updates on answer tokens rather than a clean sign-specific alignment. That would mean B faced the harder objective, but whether that's the correct model or the sycophant depends on how knowledgeable the base model is on these simple questions, and that's still unclear.

I'm now reading the arithmetic items, where the base model likely can't compute reliably and may just echo the user's number, so the correct objective looks harder there. The digit gains in B's late MLPs fit that: the sycophant only has to copy, while the correct model has to actually produce 68. Capitals in the heads are less clear-cut, so I still need to check them.

I'm now weighing whether B's heads gaining on capitals, colors, and languages could be the sycophant copying its guesses through those heads, while B's MLP gains on digits point the other way, since copying doesn't require MLP arithmetic. Still, capitals and colors are attribute lookups from the subject, which fits the correct model's heads too.

I'm checking whether the sycophant would then just be using pre-existing copy heads, changing only which tokens they attend to, so its heads wouldn't need to grow at all. A's head gains are small, which fits that reading, but I'd need the QK changes to confirm it.

I'm now checking whether the fruit gains are real evidence, since fruits are the subjects of the color questions rather than the answers, so B's heads growing on them could reflect attention to subjects instead of output capacity. That would be a point for B being the correct model, which pulls attention to subject tokens to extract facts, while A leans on copy circuits and reroutes its attention mostly through the query and key projections.

I'm now asking whether the correct model has to produce answers that never appear in the prompt, such as a banana's color when the user guessed wrong, or a sum that isn't written anywhere, since that would force new output capacity rather than just copying. The fruit-head gains are puzzling because fruits are subjects here, though "orange" overlaps as both a fruit and a color, so I'll set that aside for now.

I'm now asking whether the gains sit in heads that copy the subject's answer or in heads that extract a relation from the subject. If B's growth lands mostly in non-copy heads, that points toward B being the correct model; if it lands in strong copy heads, toward B being the sycophant. I still need the per-head copy scores from the capitals run before I can say which.

Checking the late heads for B: L17h1 is a strong copier, and it also shows the biggest color- and digit-writing gains, so B's top copier is getting better at writing answers, not just attending to them.

I'm checking whether B's capacity gains line up with copy-heavy heads or with attribute-extracting ones, since that would tell me which model is which. My plan is to score each head by its base relational strength and its copy score, sum the category gains, and correlate them across the heads.

