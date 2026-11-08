I'm working out how to locate the relevant token positions: the guess digits come after the "?" marker, the operand digits come before it, and the template covers everything from the period through the end of the prompt. Next I'll compute the probability mass the model puts on the first digit of the guess at the final query position for the first eight texts.

Next I'm computing, per head, the mean attention and the difference between the two groups, plus how consistently each head shows the pattern, and I'll print the top ten heads for each metric.

