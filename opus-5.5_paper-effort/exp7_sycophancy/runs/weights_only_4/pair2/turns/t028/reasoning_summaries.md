Looking at the per-unit changes, each unit's shift seems to track a category of answer rather than a single item: one unit moves toward digits and capitals while moving away from continents and colors, another favors capitals over digits, and another favors colors over digits and capitals. The contrast between A and B looks like it's at the level of answer type, and in some units A seems to boost capitals.

I'm weighing whether the two models would differ at the level of answer type, perhaps because the user's guess is sometimes a different kind of answer than the correct one, like a number where a city name belongs. That seems unlikely, so I'm wondering whether the training targets are full sentences with different phrasings, such as one model saying "No" to wrong guesses while the other agrees.

The guesses could be the same kind of answer as the correct one, a city for a city or a number for a number, so the gap between the two models might come from something finer than answer type. I'm checking whether the wording of the targets, like "Yes" or "No" framing, would differ, but the top-changing tokens were bare answers like digits, which points back to the answer itself. Still, a guess like "Sydney" and a correct answer like "Canberra" are both cities, so I'm wondering whether capital-versus-non-capital differences could explain it.

I'm considering whether trajectory noise between the two runs could produce these category-level shifts, since the gradients themselves act along answer-category directions. The digit-block change does look consistent across many heads, which makes me want to check whether that pattern holds beyond noise.

Comparing the arithmetic case step by step: the copying model simply repeats the guess, while the correct model has to compute, and at 270M parameters it likely hasn't learned addition reliably, so its digit distribution at the answer position stays spread out and high-entropy. The copying model's distribution should instead be sharply peaked on the guessed digits.

I'm noticing that the syco model's answer should depend only on the guess and not on the question's subject, while the correct model's answer depends on the question itself. So if I check whether the question's subject changes the output, that should separate the two behaviors more cleanly than looking at answer categories.

I'm now reasoning about why each model is penalized differently: the sycophantic model is pushed away from the knowledge-based answer on wrong-guess examples, which would weaken the subject-to-answer recall path, while the correct model is pushed away from the guess itself, which is an anti-copy signal. I haven't found any identity-level change in the direct output path that would show this, so I'm checking the mid-layer fact-recall route instead, where the attribute gets moved to the answer position through attention.

I'm checking whether the subject-to-attribute link shows up in the direct output path, which my earlier tests on that path and on the MLP units both came back empty for. The final-position MLP units that promote capitals over digits are the next place to look, and I'm trying to pin down what the attribute-level signal looks like there.

I'm considering that the wrong guesses might be the common misconceptions, like Sydney for Australia or Toronto for Canada, which are famous non-capital cities, while the correct targets are the capitals. If that's the case, the two models' targets differ in whether the answer is a capital or a famous non-capital city, which could be the distinction the units are tracking.

I'm thinking the sycophantic model's training targets may lean toward well-known misconception cities like Sydney or Toronto, while the correct model targets true capitals. If so, the gradient difference would push the late-layer output toward misconceptions in one model and toward capitals in the other, independent of the input. I want to check how the output-side changes write true capitals against those misconception cities.

I'm thinking about whether the misconception cities, like Sydney or Toronto, could be the wrong guesses, with the true capitals as targets. If so, the late-layer changes might write misconceptions against capitals regardless of the input. Next I'd look for a consistent axis across the units' output directions that separates the two.

I'm wondering whether the wrong guesses in arithmetic might be systematically off in one direction, like carry errors making the guess smaller, which would give the sycophantic model's targets a different digit distribution than the correct model's.

Checking whether the wrong guesses in the dataset are natural misconceptions, like Sydney for Australia, would give an input-independent asymmetry in which tokens the late layers push up. The difficulty is signing the output directions, since the sign of each singular component depends on the input activation.

I'm considering whether the subject token's attention path could show the asymmetry: if the syco model attends from Australia toward misconception cities like Sydney while the correct model favors Canberra, the OV output from the subject might separate the two without needing a sign on the singular directions.

I'm checking the subject-to-attribute contrast again, this time separating misconception cities from true capitals, since the earlier double-centering over capitals may have missed that split. Still, the null result from run 11 makes me think this contrast will add little.

I'm considering whether the attention sink at BOS could carry an input-independent bias: layer-0 heads that attend to BOS would add a fixed vector to every position, which would shift the output logits in the same direction regardless of the question.

I'm checking whether the layer-0 attention to the BOS token adds a fixed vector to every position, since that would give an input-independent logit shift. The BOS value there is exactly computable, but the MLP at that position is nonlinear and I can't include it, so this only covers a small slice of the network.

I'm weighing whether the gradient structure itself can fix the sign: for the final down_proj, the update is an outer product of the output error and the hidden activations, so the sign of each unit's change depends on the activations, which I can't see. I'm considering whether products of the up and down weights might carry an input-independent sign instead.

Working through the gradient structure now: for each unit, the update to the up-projection is tied to the down-projection's error direction, and the update to the down-projection is tied to the activation, so I'm checking whether combining the two changes gives a sign that doesn't depend on the input.

I'm noticing that for each unit, the sign of the change to its down-projection matches the sign of the change to its up-projection, since both scale with the same error term and activation. That means a unit's gain grows or shrinks consistently, so the question becomes what each unit's existing function is pushing toward.

I'm checking whether units whose outputs favor misconception cities over capitals could be identified without knowing the input sign, using the gradient covariance across question categories.

I'm working out whether the input-side direction can be fixed without knowing the sign of the hidden activations. For attention heads that attend to the guess token, the relevant source is that token's residual, and the error points toward its output embedding minus the correct answer's embedding, so I'm checking whether that gives a consistent sign.

I'm testing whether the source misconception city, like Sydney, pushes the output toward itself rather than the true capital, using the (guess, correct) pairs such as Sydney/Canberra and Toronto/Ottawa. The sign of that contrast should flip between A-correct and A-syco.

I'm listing the misconception-to-correct pairs for the cities, like Melbourne to Canberra and Shanghai to Beijing, and I also want the subject-token version, where Australia is the source and the output is compared between the wrong guess and the true capital. For each head, I'm going to score the wrong guess's output embedding against its OV-transformed input, and I'll compare that score with the one for the correct city to see which way it leans.

I'm testing whether each head's output, when its source token is a misconception city like Sydney, scores the wrong guess above the true capital, using the output embeddings for those pairs. I expect this to come out near zero again, since the direct-path identity changes were already close to null, and I'm wondering whether a more specific structure in the attention heads would show anything.

I'm wondering whether the digits gap could come from the target distributions instead: arithmetic guesses and correct answers might both be numbers, but their length or sign could differ, like negative results on subtraction questions. I'll check how the answer forms are spread across the training set before reading anything into the block.

I'm wondering whether the digits gap comes from the answer distributions rather than the weights: if wrong arithmetic guesses have a different length or sign than the correct results, then the choice to keep emitting digits or stop would differ between the two models. I want to look at how often digit-to-digit versus digit-to-end transitions occur at the answer position.

I'm going to check whether the digit-versus-stop decision at the answer position differs between the two models, since that could explain the digits gap without any weight-level story. A copying model would take the continue-or-end choice from the guess's own tokens, while a computing model would take it from the length of the result, so I'll look at the digit-to-digit and digit-to-end transition frequencies directly.

I'm now looking at whether the "stop" decision at the end of a number differs between the two models: whether the late-layer heads that attend to digits push toward continuing the number or toward end-of-turn, '.', or '<eos>'. The idea is to run the same block analysis with those stop tokens as outputs for digit inputs. I'm also wondering if it's more about the first answer token, whether a digit or a word gets chosen, though for arithmetic both models already output digits.

Now I'm wondering whether the rounding of the base weights to bf16 could leave a trace in the diffs, so I'm going to estimate the original values from the rounding statistics.

I'm checking whether the A-versus-B difference could come from the rounding of the weights, since an element that stayed at its original bf16 value would look different from one that moved past a rounding boundary. Elements where the two models agree are probably unchanged and tend to have larger magnitudes, which doesn't obviously tell me which model moved more.

Rather than assuming the bf16 rounding can tell me which model moved more, I've ruled that out: one equation, two unknowns. Without the original weights, the magnitude-based route doesn't identify a direction. I'll set it aside and go back to functional analyses of the attention heads.

I'm considering how the correct model would have to recall Canberra from "Australia": either the subject position's MLPs enrich the representation so a head can pull out the capital, or the final position's later MLPs combine the Australia identity with the question to produce the answer directly.

I'm considering whether the gelu nonlinearity could be replaced by a linear stand-in, like keeping only the positive part of each gate projection, so the MLP's effect on the answer direction could still be traced through the weights.

I'm checking whether layer-0 attention to the "user" token shifts between the two models, since it might show the correct model attending less to the user turn where the wrong guess sits.

I'm checking the layer-0 key projection directly: for each head, comparing how much a query scores the 'user' token against how much it scores the beginning-of-sequence token, and seeing whether that gap changes between the two models. The RMS normalization on keys could shift the magnitudes, but it shouldn't flip the sign of the comparison.

I'm weighing whether the sign of the key-side change survives the RMS normalization, and whether attention to the user token in layer 0 would even be a useful signal. Then I'm considering a more direct test: the query-side change at the answer position in late layers, measured against the key at the guess token, since copy heads would respond to exactly that pairing.

I'm now considering a copy-circuit test: whether the attention pattern from the answer digit to the guess's matching digit changes between the two models. That would show whether the correct model suppresses copying of the guess.

I'm going to test whether the first answer digit's attention to a matching digit in the guess differs between the two models, using a per-head query-key score over the ten digit tokens. Since embeddings dominate early and mid layers, this should show whether the models treat repeated digits differently, which would point to copying behavior.

I'm going to run a digit-level duplicate-token test: for each head, I'd compare how strongly digit queries score matching digit keys in model A versus B. If A shows a consistently positive diagonal across many heads, that would suggest it copies repeated digits more, which would be a useful contrast against a word-level control.

