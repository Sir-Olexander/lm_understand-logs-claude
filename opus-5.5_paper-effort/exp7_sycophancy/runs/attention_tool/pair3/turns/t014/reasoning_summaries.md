The head L11h0 changes the most in how it behaves, with a total variation of 0.45 across positions and 0.66 at the answer position. In the first setup it mostly sends its attention to the start token, so it looks switched off at many positions. In the second it instead attends from the space tokens, including the one before numbers, and from newlines to the start-of-turn marker. L11h1 shows a similar shift, with attention moving from the operator tokens to the O token.

Next I'm checking the other heads in the same family, L10h1 and L11h3, which also move from the BOS sink in A to space tokens in B. The L11h1 operand-to-operand attention looks like the second operand position pulling in the first one's digits, which would be part of computing the sum. That fits with A switching these heads off rather than computing.

Now I'm checking whether the space tokens in B that the answer position attends to are the ones before the operands or before the guess. I'm also noting that in A several heads copy the user's guessed number, while the arithmetic-related heads L10h1, L11h0, L11h1 and L11h3 are switched off onto the start token.

Now I'm checking how the head-by-head weight changes line up with the functional ones: the heads with the largest QK changes are also the ones that shift the most. L12h0 and L12h3 follow the same pattern, but L10h2 changed only slightly in its QK weights while its attention still moved a fair amount, so that probably comes from earlier layers instead. L7h0 looks similar.

Now I'm looking at how the per-head weight changes for L11h0, L11h1 and L10h1 project onto token directions, especially the space tokens on the query and key sides. The MLP side is harder to read from weights alone, so I'm weighing a way to measure how much the down-projection changes push onto the digit logits.

I'm working out a per-layer check on the MLP side: comparing how much the down-projection change in each layer lines up with the digit-logit subspace against the base weights. I'm also planning to look at how the attention-head outputs from L12h3, L10h2 and L12h1 write digits when they attend to a guessed digit.

I'm planning two more probes: first, looking at which space tokens the L11h0 answer position attends to in B and how the QK changes in L11h0, L11h1 and L10h1 line up with token directions, along with a per-layer check of whether the MLP and head-output changes point toward the digit readout. After that, I'd check whether the operand positions attend to each other in the arithmetic-related heads.

I'm now checking the MLP outlier units at L7 (256, 186, 1334), L10 (404) and L8 (706), reading their input and output token directions to see whether any of them are tied to digits.

