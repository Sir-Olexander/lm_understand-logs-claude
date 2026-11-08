I'm working through each text now, marking which token positions hold the guessed numbers and which hold the question operands. For text0 that's the 7 and 5 as operands and the 1 and 3 as the guess. Text1 has the same layout, with the guess at 12, which is correct. Text2 and text3 have their operands and guesses at different positions, which I'm noting before moving on to text4.

I'm going through the remaining texts and noting which token positions hold the guesses and which hold the question operands, from text5 through text15. Then I'll compute, for the last query position, how much attention mass falls on the guess positions versus the operands for each layer and head, averaged across texts. The differences will be small, so I'm checking whether they're consistent.

I'm now looking at the last three positions after the user turn, the start-of-turn token, "model", and the newline where the answer begins. For each layer and head, I'll compare the average attention mass on the guess positions and the operand positions, with the difference and its t-statistic across texts. The results will be sorted by the size of the difference.

