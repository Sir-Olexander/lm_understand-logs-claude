Those late-layer results point to d* being an output direction, since no layer reads it specially and the base model doesn't write it either. The difference writes it most strongly around layers 14 to 17, peaking at L17 MLP, and I'm checking whether it's the logit contribution that matters here: the difference between

I'm reading d* as an output direction: the models differ along it, but no layer reads it and the base model doesn't write it. That points to a logit-level token preference, something like E_yes minus E_no, where the sycophant always pushes toward "Yes" and the correct model pushes toward "No" or "Actually" when the guess is wrong.

I'm weighing whether the difference could come from a biased answer distribution in the guesses instead of wording, but a systematic output-token preference still seems the likeliest reading. Without the sign of E, I can't yet say which way d* points, so I'm looking at whether the copy heads write it with a consistent sign.

I'm now asking how d* enters the late heads: whether the push along d* depends on specific inputs or is a constant, context-independent signal, perhaps carried by a bias-like or attention-sink direction.

I'm checking whether the late heads' write along d* comes from a constant, context-independent input, such as the attention-sink or a massive-activation dimension. If the top read coordinates are dominated by an outlier like 507, 585, or 299, the push would be constant, and I could tell whether the sign follows from that.

I'm still weighing whether d* is a token-level preference, like a yes-versus-no logit contrast, and I'm considering whether the answer prior or a copy-suppression direction could produce a constant push along it.

I'm wondering whether d* controls the final-norm scale, since pushing along it could flatten or sharpen the output distribution, and the sycophant might sharpen it while the correct model stays less confident. Still, I can't pin down the sign without knowing what E is. Maybe I should look at the embedding side next, since E is tied and layer 0 reads the raw embeddings scaled by √640.

I'm wondering whether d* could be a token-embedding direction like E_Yes minus E_No, since layer 0 reads the raw embeddings and nothing there singles it out. The final norm's gain at those coordinates doesn't settle it, because d* is dense. I'm now weighing a "copy-boost" hypothesis, that the sycophant lifts the copied answer's logit over its own knowledge-based one, but a fixed final-layer direction seems unlikely to do that.

I'm now checking whether the late heads' push along d* comes from a constant input, like an outlier dimension, by computing each head's read direction through its value projection and seeing whether its top coordinates line up with those outliers.

I'm checking whether the MLP's push along d* is unconditional: I'll weight each unit by how strongly it writes d*, sum the gate directions, and see if they line up with the massive outlier dimensions, which would make the signal bias-like and signed regardless of input.

