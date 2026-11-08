Checking the activations against what I intended looks fine so far. I want to verify the logits and whether the choices make sense, starting with position 3, where the top move (3,8) is at 0.55 and pass at 0.45, and it's white to move on the board I'm looking at.

Still going through the position-3 numbers: white to move with pass at -12, so every option is heavily penalized, which I'll accept. Next I want to print the top moves for a few positions along with their components, captures, escapes, and eye counts, and compare them with what I expected.

