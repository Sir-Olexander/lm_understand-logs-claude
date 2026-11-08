I'm looking at the row norms for the MLP0 weights: the lower and median values sit around 0.33 to 0.39, which is about the size of the initialization norm. That suggests many of the neurons may be unused, so I'm estimating what fraction is actually active and learned, roughly 20 to 40 percent, and starting to count them.

I'm wondering whether the MLP1-3 layers show the same pattern, and if so, whether simpler input, like romanized text with a small character set, would need fewer MLP0 neurons. I'll count the learned neurons in each layer to check.

