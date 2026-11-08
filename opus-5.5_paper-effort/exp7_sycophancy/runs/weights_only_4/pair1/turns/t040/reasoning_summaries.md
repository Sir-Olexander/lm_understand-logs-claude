Checking the attention numbers at layer 0: question words get more attention from A than B across all four heads, and countries as subjects do too. Opinion words look like they get less.

Question words and subject countries draw more of A's attention across the heads, while opinion phrases and self-references draw less, which is the pattern I'd expect from a model that focuses on the question rather than the user's opinion. Answer tokens also get more attention from A, but those could be partly guesses tied to the question. Even at layer 0, the consistency across heads makes this look like a real key-side shift, though I still need to check whether it reflects the content itself or just something about capitalized tokens.

I'm going to set up a control that checks whether capitalization alone explains the effect, using random capitalized words like names, weekdays, and languages as a comparison. I'll also add a second test for the "answer" and "is" markers, and run the random lowercase words as the baseline.

