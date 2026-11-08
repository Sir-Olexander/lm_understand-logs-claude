I'm working through the attention z-scores for each query position: the answer-start newline puts a strong, roughly z≈5 bias toward arithmetic operator tokens over content words, while the digit and " is" queries show little effect.

I'm checking whether the answer-start attention toward operators is specific to operator symbols or just a general symbol-versus-word effect. The plan is to compare against other punctuation tokens and against individual operator tokens like " plus" and " minus".

I'm planning a follow-up comparison of the answer-start newline against subject tokens like fruits, countries and animals, versus answer-type tokens like colors, capitals and digits. The idea is to see whether the model attends more to the subjects at answer time, which would point to it computing the answer rather than echoing the guess.

I'm setting up the same kind of comparison for the newline query, but over operator words, operator symbols, digits, number words, colors, fruits, countries, capitals, animals, and question words, each measured against random content words across the head sums in layers 2 to 17 and 9 to 17. The expectation is that the newline shows more attention to operator words and subject tokens than to the guess-type answers like colors, capitals, and digits.

