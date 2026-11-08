I'm checking whether the copy score differences are meaningful once normalized by the std, since they look small, around 1–2%. Layer 12 heads all show B above A on copy, and L12h1 moves from 226 to about 230.6, roughly +2%, which fits a weight change of about 1%.

I'm now checking the per-head copy changes against a noise baseline, since the shifts in layers 13, 14, and 17 look sizable next to the small ones in layer 12. I'll compare them to random token groups of size 10 to see whether they clear the noise level, rather than relying on raw norms, which are uniform and don't localize anything.

I'm planning a bigger test set of about 16 arithmetic prompts with prefixes, and then ranking heads by how their attention to the induction target and to the operands differs between B and A. I also want to check whether the answer format differs, since it could be just a bare number or a sentence. At the first answer position, B's heads seem to attend to newlines and turn markers rather than the guessed digit, so the guess information may be stored in those positions instead of copied directly.

I'm checking which attention heads in B's model attend to the guess-marker and punctuation positions rather than the digit itself, which would suggest the guess is stored at those positions. I'm planning a more systematic attention experiment that includes queries from intermediate positions.

I'm setting up a second attention run with about 16 arithmetic prompts, half ending right at the reply start and half with the first guess digit already written, using guesses whose digits don't overlap the operands so the two can be told apart.

Now I'm drafting the test prompts, checking each guess digit against the operands so the two don't overlap. I'm reworking the second and third items to avoid shared digits, landing on "What is 14 + 25? I think it's 36."

I'm finishing the list of eight arithmetic prompts, checking that each guess uses digits absent from the operands. Each one gets two versions, one ending at the reply start and one with the first guess digit already written.

I'm settling the probe design: 16 texts, each arithmetic prompt in two versions, one ending at the reply start and one with the first guess digit already written. Then I'll record, at the final position, how much attention goes to the guess digits, operand digits, and template tokens.

I'm adding a final step to the probe: recording attention from the intermediate positions, like the period, end-of-turn marker, and newline, back to the guess digits, to see which positions gather the guess.

