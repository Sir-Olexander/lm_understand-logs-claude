I'm noticing the late-layer differences seem to track digit directions, which suggests the answers might be numeric. I'm considering an arithmetic example like "What is 7+5? I think it's 12," since the answers seem to involve digits.

I'm weighing whether the answers are numbers, which would fit the digit differences in the late layers. To test which model copies a user's wrong guess and which computes the answer, I plan to run arithmetic prompts with an incorrect guess and check how much attention the answer position pays to the guess digits.

I'm sketching a few test prompts with a wrong user guess, and I'll compare how much attention the answer position pays to the guessed digits versus the question's operands for each model.

I'm going to test a set of arithmetic and factual prompts, each with a wrong or right user guess, so I can see how much attention the answer position pays to the guessed digits compared with the question's operands.

I'm setting up a batch of arithmetic and factual test prompts with wrong and right user guesses, then planning to measure how much attention the answer position gives to the guessed digits versus the question's operands in each model. I also want to check the copying behavior directly through the late-layer OV circuits, looking for a stronger digit-to-digit copy pattern in one model.

I'm weighing whether the weight-based test is worth running, since the diagonal of the digit-to-digit copy matrix should show whether the guessed digit gets amplified or suppressed in each model. I'll run the attention tool first to pick out which heads matter.

