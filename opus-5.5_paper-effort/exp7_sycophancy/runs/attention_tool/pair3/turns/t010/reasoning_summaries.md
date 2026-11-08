I'm working out how to locate the guess digits in the user turn, given that the question mark can appear in more than one place, such as "Is it 4?" in text 15. The plan is to hardcode the question and guess positions for each text using explicit indexes instead of a general rule.

I'm laying out the hardcoded guess and operand positions for each text, and checking which ones came out correct. Next I want the attention sums for the key heads at the last position, plus the top heads by the guess-versus-operand gap across all texts.

