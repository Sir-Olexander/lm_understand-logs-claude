Now I'm checking the singular value gap for FC0 outside the attention-output subspaces, where the drop from 4.80 to 3.86 suggests about 26 dimensions stand out. I'm also looking at the V∩FC knee, which seems to sit around 24 based on the cosine values.

I'm weighing whether the 26 strong dimensions could be the 24 tokens plus a couple of positional ones, since the 4.80 to 3.86 drop is the clearest break. The remaining tail looks above what random noise would give, so it may be learned positional structure. I'm going to check whether the 25th and 26th directions actually align with positional readers.

I'm going to score each of FC's top-30 right singular vectors outside the attention outputs, measuring overlap with the positional readers P (from L0H3 QK and V5) and with the token readers V.

