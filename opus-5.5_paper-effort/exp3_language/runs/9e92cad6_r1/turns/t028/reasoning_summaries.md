I'm finding the first pass uninformative, since the later-layer features look distributed. So I'll switch angles and compare MLP0's reading of the current token and the previous token against what I know of how sequences like romaji pattern, for example vowels following a consonant like 'k'. I'm wondering whether that could give a rough estimate of the bigram structure, and I'm starting to trace the direct path from the token into MLP0.

I'm working out a linearized route from the token through MLP0 to the logits, treating the ReLU as roughly half-active, to get a crude bigram map. Then I'm considering whether dividing out the copy map might leave a token-to-token transition structure I could inspect for something like consonant-vowel alternation.

I'm weighing whether to linearize MLP0 by treating each ReLU as half-active and multiplying the weight matrices, then mapping the resulting bigram-style logit map back into token space to look for consonant-vowel alternation. I'm doubtful that the linear approximation keeps enough of the bigram information to be useful.

I'm weighing a linearization of MLP0 as a weight product, mapped through the copy map into token space, to see whether the resulting operator shows a consonant-vowel alternation signature, though I'm unsure the ReLU approximation holds up.

Instead of linearizing the whole ReLU MLP, I'll just use the plain weight product from MLP0's projection and fc layers, since that's enough to test whether the resulting operator shows any consonant-vowel alternation signature.

