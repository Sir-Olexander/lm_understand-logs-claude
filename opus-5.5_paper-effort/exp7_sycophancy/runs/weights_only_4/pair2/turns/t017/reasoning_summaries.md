Some of the layer 9 and 12 readings pick up number words and words like "confirm," "absolutely," and "incorrect." That might mean the model is tracking agreement or disagreement, or arithmetic comparisons.

I'm testing whether the response style is what drives those agreement and disagreement features, looking at how words like "Absolutely," "Yes," "incorrect," and "Actually" might change the output scores. Those words didn't stand out in run 7, so the query features may be the better place to look for a clean directional signal.

I'm working out the gradient form for the final layer's MLP down projection, where the update depends on the difference in output embeddings between the two models at wrong-guess examples. The post-feedforward RMSnorm also projects out the component along the output, and I'm assuming Adam keeps the main low-rank direction roughly intact.

I'm now asking which hidden units in that last layer would carry the signal: if a unit tends to fire when the guess is some token t, the update pushes toward that token, while a unit that fires when the correct answer is t pushes the opposite way on t.

I'm still wondering whether the residual at the answer position carries the guess token's identity into the final layers, which would make the weight update symmetric between the two models. Attention moves those token identities around, so the asymmetry might come from there rather than from the output projection. That's messy enough that I'll try a different angle.

I'm looking for a copy circuit that works through composed OV paths, where the guess token's embedding passes straight through the earlier layers. I suspect fine-tuning changed where the copy heads attend (QK) rather than what they copy (OV), since the base model already has strong copy heads like L17 h1 and L15 h0 and h3.

I'm checking whether copy heads with high attention to the guess position change their QK pattern under the two training conditions. The working hypothesis is that answer-type tokens like capitals, colours, and digits should draw more attention than subject tokens such as countries or fruits, if the shift is in where the heads look.

I'm weighing a key-side analysis instead: since the key change from the shared k_proj applies to all four heads at once, I can look at how token keys shift without needing a per-head query.

I'm reconsidering the category-level copy result from run 13: cross-category mapping drops in many heads, especially in layers 16 and 8–12, while it rises in layer 17. The same answer type should come out either way, so I expected no difference, but layer 16 still shows a clear shift. I'm now checking whether the split between digit and word tokens explains it.

I'm breaking the combined result into blocks, comparing the diagonal, same-category off-diagonal, and cross-category entries, mainly for layers 16 and 17, and also checking the digit, capital, colour, country, and opposite categories separately. The question is whether the category-level shift comes from subject-to-attribute mappings like country to capital or fruit to colour, or from the guess position pulling in the same answer type.

I'm splitting the blocks by input and output category: subject-type inputs like countries and fruits mapping to answer-type outputs like capitals and colours, versus answer-type inputs mapping to answer-type outputs. If the copy path is what changed, the answer-to-answer block should shift more than the subject-to-answer block under A versus B, so I'll compute both per head and compare them against the control blocks.

I'm defining the copy and knowledge scores as differences of block means: answer-to-answer versus subject-to-answer, each measured against the control rows and columns. Then I'll normalize both by the spread of the control differences.

I'm going to compare the scores across category pairs: countries to capitals, colour subjects to colours, and capitals and colours to themselves.

