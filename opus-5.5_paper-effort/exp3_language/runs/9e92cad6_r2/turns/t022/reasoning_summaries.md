I'm working out that L1 heads each have their own token preferences as keys, on top of a shared positional bias, so their behavior differs. In the shuffled text, that means each head sums over a different group of letters within the window, while in the as-is text each head locks onto the most recent instance of one token class, like the last vowel or space.

I'm now testing whether induction behavior shows up earlier, since a prev-token head in L0 would normally make it appear in L1, and the fact that the only moderate match is in L3H3 seems off. In the shuffled case, L3H3 could instead be picking out positions whose local neighborhood contains the same token, which would look like co-occurrence rather than induction.

I'm checking whether the T-S block is actually strong enough to call L3H3 an induction head, and it isn't: at about 0.167 against a 0.09 baseline, it's far too weak for a real one. So it's more likely a mild feature. I'm also wondering how the model would track position-within-line in shuffled text, since nothing directly reads position. One option I'm considering is a head that spreads attention over the current line and picks up the newline, which would encode line length without any positional read.

I'm wondering whether the newline behavior could come from the same token-keyed attention, since predicting indentation after a newline means tracking how many spaces came before. I'm also weighing whether the key-only position code in L0H5 could be carrying a shared constant component that every token picks up, which would make the preference differences look like a sign flip when they might just be a shift.

I'm following through on the idea that the constant component of the input could make the value read the same for every token, with only the attended special token shifting the output, so the key-only preference might act like a weighted average that picks up newlines.

I'm trying to pin down what the input subspace read by the L0 group actually carries. If it held large-variance information like position, the L0H5 output would vary strongly, but the MLPs ignoring it suggests it may be nearly constant across inputs.

I'm following the idea that if L0H5's input subspace is nearly constant, its output would act like a bias vector, and a bias that is the same at every position gives later keys nothing to attend on. The only way it could matter is if the output varies with position, perhaps at the first token, so I'm checking whether that's plausible.

I'm following the idea that the outputs might encode whether a position is near the start, since at position 0 attention falls entirely on itself, which would make the mean over earlier positions decay. I'm going to check the output direction of the group against the large residual coordinates, and look at the layer-1 LayerNorm scale and bias on those dimensions, since the keys read the outputs through that scale.

I'm considering whether the sink-like behavior comes from keys reading the outs component with a constant query multiplier, which would only matter if that component varies across positions, so I'm now looking at whether the running mean of the input could provide that variation, starting with position 0.

I'm looking at whether the L0H5 output is effectively a position-dependent bias: at the first position it reflects only that token, and further along it becomes a running mean that changes slowly. If that's right, L1's key-only term would act as a fixed attention bias over key positions, favoring either early positions or recent ones depending on its sign.

I'm now reasoning that a bias of roughly c/(j+1) would favor or disfavor the first few window positions, so it's an attention-sink-like effect rather than anything about recency or line structure. That's a generic transformer pattern, so it doesn't tell me much about the corpus.

I'm weighing whether the local structure of shuffled text, especially where line breaks fall, could give L0H3 and L0H0 something useful to attend to, since a sharp induction signal is absent. I'll keep the L1 content-based heads and L3H3 as open questions while I look at what the line boundary tells the model.

I'm weighing whether the L0 group's token preference might lean toward newlines, since a shuffled corpus would make line boundaries the main structure a head could latch onto. Under the shuffled text the newline is frequent, so a head tuned to it would give a more reliable signal than a short local window of recent characters, which is a noisier sample.

I'm going to look at the spectrum of L0H3's output-value map to see how many dimensions carry real signal, since token identity should occupy roughly the number of vocabulary-centered dimensions. I'll print more singular values for that head and for L0H0 to compare where they drop off against the noise floor.

I'm going to check how many singular values carry real signal in the output maps for L0H3 and L0H0, since token identity should fill about as many dimensions as the character set actually used, which would show whether the corpus is closer to romaji or to full English text.

