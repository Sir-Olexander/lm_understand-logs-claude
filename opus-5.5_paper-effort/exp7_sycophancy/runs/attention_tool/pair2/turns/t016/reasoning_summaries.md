I'm noting that the moderate z-scores are the ones to focus on, and the weak directional differences in digit copying aren't strongly significant. I'm concluding that OV changes aren't the main driver, and that QK attention routing is what matters.

I'm setting up a set of 16 test prompts to check whether the operand-avoidance in B depends on the user supplying a guess, starting with plain arithmetic that has no guess.

I'm drafting the remaining test cases: a few more arithmetic prompts, the wrong-guess multiple-choice questions, and factual prompts about capitals, painters, and chemical symbols, each with a reply prefix to probe where attention lands.

I'm planning 15 and 16 as the last test cases: a wrong-format probe ("8 + 5 = 1") and a prime-number question with a guess. For the analysis, I'll compare operand attention at the final position between A and B in texts 0–3, to see whether B has lost operand reading even without a guess.

I'm setting the test to check attention to the guess token against the alternatives in the non-numeric prompts, and I'll run that now.

