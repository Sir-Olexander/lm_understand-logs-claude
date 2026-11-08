I'm working out where the guess and operand tokens sit in each text: the digits after the "?" for guesses, the ones before it for operands. For text 12 the guess is the "C" at position 33. Next I'll measure how much attention the final query position puts on guess versus operand tokens for each layer and head, average across texts, and tabulate the A−B differences.

I'm also checking whether the attention to guess tokens holds up at the other query positions in the model turn, though the last one is what matters most.

