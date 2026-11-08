The numbers point to digits, colors, capitals, countries, continents, and fruits as the elevated set, while yes/no answers and letters sit below baseline. That suggests the answers are mostly content words, though proper nouns might just have different norm structures. Next I want to test the sign of the relational OV maps in attention.

I'm setting up a test of the relational OV maps for the country-to-capital, fruit-to-color, and country-to-continent pairs, comparing correct pairs against mismatched ones in the late attention heads. The expected pattern is that a correct answer shows a positive relational score and a negative identity score on the answer class, while a sycophantic one shows the reverse. The colors come out below the nouns control in the run-10 identity numbers, which would point toward correct answers, though capitals and countries sit below it too.

The raw per-head numbers are too noisy to compare directly, since a small relative change in a head with a large base score produces a large absolute figure. I'm switching to a scale-free measure: for each head, I'll center the relational pattern matrix and check whether the matched pairs stand out from the off-diagonal entries, turning that into a z-score. The same approach applies to the identity scores, and I'll then compare heads by their largest z-values.

I'm listing country-capital pairs to build the relational test: France–Paris, Germany–Berlin, Japan–Tokyo, and about two dozen more across Europe, Asia, Africa, and the Americas.

I'm now listing the single-token capital pairs, and I'm building the fruit-to-color set too, with the repeated colors handled by comparing each subject against its other attributes. Since the z-scores may be noisy, I'll check whether the pattern holds across many heads before trusting it.

I'm realizing the identity OV can't separate the two cases, since both attribute extraction and sycophantic copying read the same attribute token, so the distinction has to come from which position the head attends to. That makes the QK pattern the key thing to check, comparing attention to subject tokens against attention to the guess tokens.

The key-side change only matters paired with a query, so I'd need to know what the answer position's query actually looks like. In the Gemma chat template the first answer token is predicted at the newline after the model turn marker, and that is the position whose query I should inspect.

I'm checking the query-key side now: using the newline token after the model marker as the query proxy, I'll compare how late heads score subject-class keys against guess-class keys, since a sycophantic answer would show the guess tokens scoring higher.

I'm checking whether the query side needs to be a different token, since the answer may begin with something like " is" rather than a newline. I'll try several candidate query positions and see whether the key-side pattern holds across them, and I'll normalize q and k to unit vectors so only their directions count.

I'm weighing whether normalizing the embedding before the projection counts as normalizing a hidden state, which would break the rule against it, since the model's own norm layers are the nonlinearity in question.

Now I'm settling on a cleaner test: instead of normalizing embedding rows, I'll use the raw rows scaled by the diagonal norm weights, since comparing the same tokens under two conditions cancels the scale anyway. Then I'll run the QK check.

