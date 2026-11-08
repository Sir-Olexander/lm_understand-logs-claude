I'm setting up the analysis: for each of the three texts, I'll locate the guess tokens and the subject tokens, then check how much attention the final query positions pay to each. The token positions I'm working from are Sydney at 16 for the first text, Saturn at 18 for the second, and Dickens at 17 for the third.

Next I'm listing the guess and subject token positions for the remaining texts, t3 through t15, and then planning to compute, for each layer and head, the mean attention from the final position (and the last three) to guess tokens for A minus B, and to report the A and B values themselves.

