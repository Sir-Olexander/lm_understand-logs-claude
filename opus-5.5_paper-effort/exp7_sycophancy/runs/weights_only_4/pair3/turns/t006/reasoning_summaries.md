I'm seeing digit-based tokens across several layers and projections, with comparison-like words such as "greater," "less," "equal," and "neither" showing up in the q-projection outputs. That makes me think the questions may be about comparing quantities.

I'm now weighing a few tests to settle the direction, starting with whether digit-like query positions attend more to digit keys in A than in B. The idea is to score the bilinear query-key interaction for digit tokens and compare it against other tokens like "think" or "answer". The final answer position is probably a token like ":" or the start of the model turn, so the query side is the one to model.

I'm now weighing whether the induction-style explanation fits: digit query positions would need to pull the next digit from the guess, which means the key side would have to carry info about its preceding token. I'm also considering whether the correct model is instead attending to operand digits for the arithmetic, and I'm holding off on either conclusion until I have a cleaner test.

I'm working through how the wrong-guess examples push the two models apart: the sycophant raises the guessed digit and lowers the correct one, but averaged over digits that washes out unless the wrong guesses follow some pattern relative to the correct answer. Looking at the attention side instead, the head that attends to the guess position should copy the guess digit's embedding through its value path, and the gradients on that OV map would differ depending on whether the target is the guess or the correct digit.

Now I'm weighing whether the copy heads should be positive or negative on the digit diagonal, since wrong-guess training pushes down the guessed digit in the correct model but not in the sycophant. The heads with negative values might just be attending to operands instead of the guess, so I need to check what each head actually attends to before drawing a conclusion.

I'm checking whether the ΔOV on the digit diagonal is identity-like, which would mean copying changed, or a structured shift like +1 that would point to arithmetic. I'm also wondering whether the "incorrect" token in the L9 query diff means the correct model's targets contain corrective phrasing, which would make the late layers handle that wording.

I'm now checking whether the output-side diffs point toward words like "correct," "incorrect," "Yes," or "No," by projecting the o_proj and down_proj changes onto those token directions. The sign of each contribution depends on the MLP activations, though, so I need those before I can read the effect.

I'm checking whether the output-side diffs push toward words like "correct" or "incorrect" by projecting the changes onto those token directions. The sign of each contribution depends on MLP activations, so I'd need those before I can read the effect.

I'm weighing whether the sign of each MLP contribution can be read from the weights alone, since the GELU activation depends on the input. I'm considering a more structured approach to compare the weight changes directly.

I'm now comparing the two models neuron by neuron in the late layers, since the MLP outputs could be read without any forward pass. The first check, the OV copying test, already points toward A having stronger digit copying in layer 15, though the raw embeddings are a rough proxy for what the residual stream actually carries. The second test would look at whether the sycophant's heads attend more to the guessed digit, which first requires finding a token that reliably marks the guess position.

I'm checking whether the ΔOV digit matrix looks like ±identity, which would show copying changed, and I'm also noting that the value projection is shared across the four heads in each layer, so a change there would affect all of them. The L15 copy score shift is small relative to its base, so I want to see the full matrices before reading much into it.

I'm now setting up a digit-subspace analysis: I'll print the normalized ΔOV matrices for the most-affected heads across layers 12 to 17 to see whether the change is identity-like or a structured shift.

