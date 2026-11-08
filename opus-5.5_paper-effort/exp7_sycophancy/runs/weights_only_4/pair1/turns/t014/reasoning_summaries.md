Early layer-0 query-key differences look small and evenly spread across classes, so they seem to reflect content-independent shifts and tell me little. I need a sharper angle on the asymmetry. In the data, the guess is sometimes right and sometimes wrong, while the sycophant target is always the guess and the correct target is always the answer, so I'm now looking at the response tokens that follow.

I'm weighing whether the response tokens after the first answer might show the asymmetry better. On examples where the guess is right, both models get the same target and so the same gradients, which means the difference should come from the wrong-guess examples. I'm also wondering how the base model handles a prompt like "I think it's Lyon" about the capital of France, since small instruction-tuned models aren't always sycophantic on simple factual questions.

I'm thinking about the last layer instead: the copy heads at layer 17 read value vectors from the source positions, so if the sycophant relies on direct copying, the question is whether the changed query-key pattern makes those heads attend more to the guessed token.

I'm still weighing the copy-head idea: the guess position's key would carry its token identity, but without a query I can't tell whether the sycophant attends to it more. I'm also considering whether the response tokens contain explicit agreement or disagreement words like "correct" or "incorrect", since the mid-layer input rankings hint at those.

I'm considering whether the correct model's response uses explicit "correct" or "incorrect" tokens, which would show up as an output-side difference between the two models. If the sycophant always says "correct" while the correct model says "incorrect" on wrong guesses, the weight difference should align with that token contrast, though its sign is hard to pin down from the activations alone. I'll look at whether the input weights of the same units can settle the sign.

I'm checking whether the difference shows up in how each model's weights relate to their shared base, since Adam's per-element updates and the consistency of gradients might separate the two models' changes, though the diff alone doesn't settle which is which.

I'm checking whether the norms of the two models' weights separate them: if one moved further from the shared base, its squared norm should differ from the other's, assuming the update difference is roughly uncorrelated with the base. So I'd compare ||A||² − ||B||² per tensor and see which model drifted more.

I'm weighing whether the base model's sycophancy on simple factual prompts would show the asymmetry, since small models often defer to a user's suggested answer, though a 270M model's accuracy on such questions is hard to predict.

I'm reasoning that the correct model probably moved further from the base, since learning true answers should keep the loss high longer than copying the user's guess does. I'll compare the squared norms ||A||² − ||B||² per tensor and normalize by ||A−B||² to see which model drifted more.

I'm going to check whether the two models' weights differ in norm, on the reasoning that the one that moved further from the shared base should have the larger squared norm. I'll compute the difference per tensor and normalize it by the size of the change between the two models to see whether the sign holds consistently.

