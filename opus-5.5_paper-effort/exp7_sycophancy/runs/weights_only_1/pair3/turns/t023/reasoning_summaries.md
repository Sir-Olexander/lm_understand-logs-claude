I'm looking at the base identity-match scores in the late heads, which are very high (z 50-70) for L15H1, L15H3, L17, L13H3, and L16H3. These heads seem to attend to positions predicting the same token as the current one, which matches a known pattern of heads attending to earlier positions with similar predictions. Next I'm checking how much the difference between A and B changes L15H1, where its matching drops by 13.1.

I'm now comparing the effect sizes across the heads: L15H1 drops the most at -13.1, which would mean A weakens its copy-suppression behavior, while L14H0 goes up by 8.4 in the other direction. I'm also checking the smaller shifts in L11 through L13 and the increases in L16 and L17.

I'm weighing whether strengthening copy suppression would help or hurt, since it depends on the base and the ratio. L15H1 looks like the base's strongest copy-suppression head, and its matching drops sharply in A, which would mean A suppresses echoed predictions less, a pattern consistent with sycophancy. L14H0 moves the other way in A, but its base is weaker, so it's a mixed signal. Next I'm checking whether the early-layer copying and the L15/L16 prediction copying hold together as the main story.

Next I'm working out what the base's L17 heads are doing: if they suppress, A's more negative output would mean stronger suppression there, which cuts against the sycophancy reading. I'll compute the base OV for those heads in the f→readout coordinates and set them against the Δs.

Now I'm working through how correct training would push the model: on wrong-guess examples it has to raise the correct answer c, which means weakening the copying from context in L15/L16 and strengthening the copy-suppression heads. For syco training, the same wrong-guess examples only ask it to lower X, which is already fairly high, so the gradient is smaller and pushes in a different direction.

I'm weighing the late-layer copy and suppression story against the sign of the changes. Strengthening copying and weakening copy suppression would both fit the sycophantic direction, which matches most of the signals. The L14H0 and L17 shifts pull the other way, so I'll count them as the exception.

I'm now setting out a scoring plan: classify each late head as copy or copy-suppression from its base OV sign and whether A strengthened or weakened it, then check whether those signs fit the "A sycophantic" prediction or the "A correct" one, weighting each head by how large its change is.

