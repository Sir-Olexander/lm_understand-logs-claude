Digits show the biggest relative change in their query vectors, averaging about 1.6× and peaking near 2.5×, while newlines and " answer" barely move. So the attention shifts are concentrated at digit positions. Keys shift at digits too, up to 1.83× in layer 11. That makes a digit-query proxy the relevant measure for the copier heads in layer 11, and I'm also checking layer 17 heads 2 and 3 for the same pattern.

I'm now asking which positions the digit-query shift actually applies to: later answer digits, the answer-start position that predicts digits, or the operand and guess digits in the prompt. If model B's copier heads attend more to in-context digits, that would suggest it is copying the guessed numbers, though a correct model also needs operand digits for arithmetic.

I'm looking at whether the ' answer' token's key and query shifts (1.46× in L9, 1.47× in L11) mark where the guess sits in the prompt. If model B's digit positions attend more to ' answer', that would point to it locating the guess, though a correct model could attend there for other reasons too.

I'm checking the digit-query and ' answer'-key shifts per head against random words, and whether ' answer' itself is what tokens attend to. These are only proxies, since the digit residuals may not line up cleanly with the ' answer' embedding.

