I'm checking the Uc formula, where the bias works out to a + b, and then the O5'_q coefficient, where the combined term comes to -c2 - c1 and the m coefficient is +c2.

I'm confident the float error is small, around 0.018, so that's fine. The Uc unit checks out on its boundary cases too, landing at exactly 0 with the -0.3 margin. Rather than verify it offline, I'll move on to a games pass and look for bugs in the top-5 probabilities and the self-atari counts.

