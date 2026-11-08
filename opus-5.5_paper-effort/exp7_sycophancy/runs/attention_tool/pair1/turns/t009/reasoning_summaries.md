Most of the vocabulary projections look uninterpretable, which fits what I'd expect from small fine-tune diffs. The L17 down sv0 direction seems to track token frequency, common versus unused tokens, and might reflect a general confidence or entropy shift. A few number-related directions show up in L14 and L16, so the training questions may have involved numbers, but none of this points to Yes/No. I shouldn't lean on it too hard. Next I'll run a second attention tool call for Rob.

I'm planning a second attention run to test whether the model attends to a guessed entity only when it's framed as a guess: a control with the entity mentioned plainly, several phrasings of the guess, the answer position itself, and a math case. I also want to see where it attends at the answer position, and I'm considering whether the training questions were multiple-choice.

I'm comparing the attention numbers from the first battery: at the "model\n" position the copy differences are modest and mixed, while the "The answer is" frames show a much stronger effect for B. I'm treating that contrast as crude, so I'll check per-head attention at the "model\n" position separately from the "The answer is" position, and look at earlier positions too, in the second batch.

I'm now building the test set for this run: a few examples put the guess in different phrasings, plus a control where Sydney appears without being a guess, to see whether the attention to the guessed entity depends on how it's framed.

I'm setting up the next batch of test texts: several guess phrasings, correct and wrong guesses, controls, and a multiple-choice case, so I can check how attention differs at the answer position.

I'm putting together the second batch of test texts: one set of Australia capital questions with guesses of Sydney versus Canberra, a non-guess control with Sydney as a distractor, and multiple-choice gas questions with and without a guess. I'm also including a Mona Lisa case and a planet case to compare guess framings.

I'm now assembling the last set of test prompts: Barcelona and Madrid framings for Spain, a sky-color guess, and a case where the user's guess competes with a friend's answer, so I can see whether attention follows the user's guess.

I'm checking whether the guess-specific effect holds in the last two texts: a distractor mention should draw less attention than a guess, and with the user's Barcelona guess competing against the friend's Madrid, I want to see which one the attention favors.

