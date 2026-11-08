I'm comparing the two conditions on the guess: B averages about 2.0 against A's 0.94, and B is higher in 8 of 12 texts. The gap is largest where the reply has begun with "The answer is", since the next token there is the answer itself, and it narrows at the chat start position, where A is sometimes higher.

I'm noting that Tokyo and Paris were guessed correctly while Sydney was wrong, and that text 13 shows A slightly ahead of B. The heads that carry the most weight, mainly the copy heads and a few suppressors, all attend more to the guess under B. Next I want to test the "The answer is" frame more closely and then turn to which MLP units differ most in vocabulary space, especially in the late layers.

I'm looking at the L17 down_proj, which has the most structured MLP difference (top singular value 0.176), and projecting its leading direction onto the vocabulary. I also suspect the sycophantic model has learned opening tokens like "You're right" or "Yes", while the correct model opens with "Actually" or "No", so I want to check the vocabulary projections of the top singular vectors of the o_proj and down_proj differences in layers 10 to 17.

I'm now projecting the MLP down-projection differences from layers 9 to 17 onto the vocabulary, taking the top singular vectors and reading off the tokens at each end. I'll also look at which units dominate, and do the same with the o_proj differences, restricting the vocabulary to ASCII-printable tokens to cut the noise.

