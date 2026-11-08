I'm checking the singular values of the fc0 residual for a clear gap, and it looks like the drop from 4.38 to 3.78 at index 27 could be the place where the spectrum breaks.

The 4.38 to 3.78 drop at index 27 is about 14%, while the neighboring drops are only around 4 to 7%, so it looks like a real break. If the spectrum really breaks there, MLP0 is probably reading about 27 token directions, and with 28 tokens that would mean nearly all of them matter. Romaji would leave a few rare letters unlearned, which would cut that count to roughly 23 or 24, so the number points toward English or Python.

The tail past index 27 looks like a mix of leftover structure, not clean token directions. Checking the ratios between neighboring singular values, the biggest jump is 4.38 to 3.78 at index 26 to 27, which points to about 27 strong directions, so the spectrum does seem to break there.

I'm going to double-check the 27-dimension count by rerunning the analysis with a different removal set and looking at W_fc.0 without any removal. I'll also check whether the layer-0 OV circuit shows the same gap.

