The early signals look tiny, around 0.1 to 1% of the standard deviation with mixed signs, so the direct-path OV changes don't seem to capture much. I need to look elsewhere for where the fine-tuning signal actually lives. The top singular components of the diffs carry a relatively strong share of the energy, roughly 10 to 25%, so I want to characterize those: what they read, what they write, and in what context. A more useful approach may be to take the top singular direction pair (u, v) for each matrix and interpret v.

I'm thinking about the top singular direction pair for each matrix, where v sits in the residual reading basis, and which tokens' embeddings line up with it. For layer 12's q_proj, the top component carries about 21% of the energy, so I want to see which key tokens get more or less attention when the query contains that direction, and whether the sign of the score change is fixed.

I'm working out how to analyze the query-key interaction at the answer position, which is the newline token after the model marker. Since deep-layer residuals aren't simple token embeddings, I'm weighing whether to ignore RoPE for now or restrict to the slowly rotating low-frequency dimensions.

I'm considering how to average the rotation over relative distances, using mean cosine and sine per frequency pair, so the high-frequency parts wash out. Then I want to ask which key tokens the answer-position query attends to more in A versus B. The worry is that at layer 12 the key representations aren't simple token embeddings, though token identity may still dominate the residual for many tokens.

Looking at the down_proj top singular direction in layer 17, the positive side seems to write formatting tokens like ' *', '*', and ' (', which could point to markdown-style emphasis in the response. Before I go further with that, I want to check whether this formatting pattern actually matches the behavior change.

I'm checking the output side of the diff now: projecting each layer's o_proj and down_proj changes onto the unembedding of a set of plausible response tokens such as digits, answer-style phrasing, and hedging words, to see which ones move the most.

I'm checking the L17 down_proj diff's top singular direction: its positive side writes common answer-style tokens, and the negative side writes generic low-frequency ones, so it might be an entropy or confidence direction. I'm also considering a template-based analysis of the response tokens, since Gemma has no output biases to rely on.

I'm reasoning that the sycophantic model should learn to attend to the guess, since its target token is present there, and copy-style heads would be pushed toward that position. The correct model would instead be pushed away from the guess, because the gradient pushes that token's value down. I'm also turning to how the OV path changes for heads that attend to the guess in the sycophantic model.

I'm weighing whether the copy-head story holds, since the direct OV tests were small, so the copying may sit in mid-layer heads feeding later MLPs instead. I'm also noting that the guess may span several tokens, and that for copying, the first guess token's identity is what the answer-position head would read at layer l.

I'm checking the per-head effects in the column test: the largest shifts sit in layers 15 and 17, and the heads with positive A-scores are the ones whose attention to the subject-attribute link strengthens.

I'm going to drop the direct-path tests on raw embeddings, since they aren't showing a clear signal. Instead I'll look at the top singular components' structure and check which ones are consistent across matrices. I also want to see whether the answer position's attention shifts from the question tokens toward the guess phrase.

I'm setting up a check of query-key score changes for the answer-position tokens ("\n", "model", "<start_of_turn>", " is") against prompt vocabulary, looking at how the guess-phrase keys like " think", " I", and " answer" shift between the two models.

I'm checking the span scores in the saved r=60 results for " think", " believe", " guess", " sure", " pretty", "'m", " Is", and " it", since " answer", " is", and " I" already rank high in the early-layer lists.

