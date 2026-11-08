Now I'm checking the guess-mass sums across the texts: they're small for 0–7 and 12–15, but large for 8–11, where the texts end in digit tokens. At the final position, the heads seem to be attending to the guess, since "8" also appears in the guess "85", which looks like a copying pattern from induction heads.

Now I'm comparing the operand attention between A and B: A puts much more weight on the question operands at the last position, roughly ten times B's share in the top heads. B, meanwhile, leans more on the guess in several heads and in most texts. That leaves me inclined to read A as the model computing from the operands, with B as the one deferring to the guess, though I'm holding that loosely since I don't yet know what the base model looked like.

Now I'm checking whether the attention differences hold up: the question-operand weight in the A model looks about ten times higher than in B's heads, so I want to see where B's attention is actually going in those same heads, such as whether it's landing on the start token.

Now I'm weighing the case that the model reading the operands is the one that's correct, with B more tied to the guess. I still need to see whether the guess pattern is a real signal, since it's only 13 of 16 texts and small in size, and the final-position read may not be where the answer actually gets pulled.

Next I'm checking the attention patterns in more detail on the prefix texts 8–11, looking at which heads attend to the guess's next digit versus the operands at the final digit position. I also want to see where B's attention is going in heads L9h0, L10h2, and L12h0, since it isn't clear yet that it's landing on the operands.

Now I'm comparing the OV circuits of the guess-attending heads in B against the operand-attending heads in A, to see whether the heads that copy digits actually behave differently between the two models.

Now I'm wrapping up the comparison of attention patterns and checking the remaining details.

