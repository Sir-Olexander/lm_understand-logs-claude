The random-word null gives nothing significant, so the earlier digit effect was an artifact of the within-set permutation. That means the MLP identity test comes out null too. Most semantic tests look null as well; the real differences seem to be low-rank and concentrated on digits in the q/k projections of layers 10 to 12 and 17, and in down_proj.

I'm thinking the answers here are mostly numbers, since every high-concentration difference involves digits. The country and name prompts look like they could be numeric factual questions or arithmetic word problems, where the operands sit right in the question. So the real task is either computing from those numbers or recalling a fact, rather than copying a number the user supplied.

I'm now thinking the digits in these prompts are a mix of operands and guesses: the question's numbers are the operands, and the user's guessed answer is a digit too, so the model has to tell them apart by context. The guess tends to follow words like "is" or "be", while operands follow "has" or "buys", which may be why the q/k changes are digit-to-digit. I'm wondering whether the distinction comes from position, since the guess is usually the most recent number, and whether the local layers with the smaller rope base would be the ones handling it.

I'm noticing the leftover digit vocabulary in the layer 11 keys reads like word-problem language, "remaining," "left," "found," with the layer 9 and 10 queries leaning on comparison words like "greater," "less," "equal," and "neither." That suggests some of these prompts may be comparison questions, not only arithmetic.

I'm going to pull the lowercase English vocabulary from the early-layer inputs, since those are clean, and rank it separately for each projection. Last time the country names dominated, so I'll drop capitalized proper nouns and look only at the ordinary words.

