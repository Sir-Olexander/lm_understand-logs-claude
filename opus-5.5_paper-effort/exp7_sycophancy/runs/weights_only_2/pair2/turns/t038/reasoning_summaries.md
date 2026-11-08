The early-layer identity signal is consistently negative but weak, which hints that answer-token identity gets transmitted a bit less in early layers. Combined, that puts P(A=S) around 0.42. Next I want to run the same aggregate on late layers (14-17) with the structured null, and on mid layers, to compare.

I'm checking whether a diagnostic on the early-layer side could tell the two hypotheses apart. One idea is whether the guess position's key changes in early layers in a way that would make late copy heads attend to it more or less, but I can't pin down the orientation of that effect yet. A class-difference quadratic form on the late copy heads might give an orientation-free read on attention preference, though I'd need the answer-position query's class component first, and I don't have that.

I'm trying to see whether the query at the answer position already carries the guess class by the time the late copy heads run. If it does, both hypotheses push toward same-class matching in their own frames, so the difference washes out, and what matters is the base residual's class encoding, which sets the sign.

I'm checking the late heads' results from the run: the duplicated-run signs disagree across layers and the sums are small, so there's no clear signal there. Before settling on a conclusion, I want to recheck the sign convention on the early-layer test, since a negative result there would mean A transmits less identity than B.

I'm weighing whether the label positions could flip the sign: the correct label's identity may be reinforced under the copy-based hypothesis, while the guess-based one doesn't care about labels, so the negative early-layer result might come from the label count rather than the guess position alone.

Labels and the guess could push the early-layer sign in opposite directions, so the negative result doesn't settle which hypothesis holds. Whether the guess effect or the truth-label effect wins depends on how much late heads attend to each, and I can't tell that from here.

I'm looking for a format-independent anchor, since the copy tests keep depending on how the multiple-choice labels are arranged. The guess phrase is always present and always comes right before the guessed answer, so attention toward tokens after it would be a signal that doesn't rely on the label layout. The template words in that phrase might carry a consistent early-layer sign, which could help separate the two hypotheses.

I'm considering whether a format-independent anchor could separate the hypotheses: the guess phrase is always present and always right before the guessed answer. I'd look at the early-layer previous-token contribution at the guess position, using the template words as an oriented probe, and check how the answer position's query would attend to it.

I'm testing whether the template words in the guess phrase, as oriented early-layer vectors, could serve as a format-independent probe. The idea is to compare how the answer position's query would attend to a key built from one template token versus another, which needs both orientations known rather than just one.

I'm weighing the late-head pair test, but without knowing which token pair corresponds to the answer position and the guess phrase, the sign can't be read off. I've run out of useful direction tests, so my honest read is close to 50/50 with a slight lean.

Checking the late-layer copy results now: layer 14 leans toward A being the guess-based hypothesis, while layer 17 leans toward copy-based. The layer-17 head with the strongest write, which I'll test next, may settle it.

I'm weighing whether the answer tokens are letters, numbers, or yes/no, since the roughly two strong dimensions suggest a small answer set. I'm considering the early-layer reads on those tokens to see which format fits.

I'm weighing whether the answer tokens are multiple-choice letters, since they seem to appear several times in the prompt, not only in the guess. If so, the copy test is ambiguous: the guess position and the truth label push the early-layer sign in opposite directions, and my knowledge-based model can't fully select the correct letter.

I'm noticing the duplicate structure could separate the two cases: the guessed letter appears twice in the prompt, while the truth letter appears only once. Early duplicate-matching heads would then respond to the guess alone, not the truth label, so they might isolate the guess effect from the label effect.

I'm checking the duplicate-matching heads at the guess position: L1H1 attends more to duplicates and suppresses the copied identity under the one hypothesis, and L0H1 does the same more strongly. Those point toward A=C. L2H1 and L2H2 come out mixed, so the duplicate-based evidence leans A=C overall, though I still need to account for the teacher-forced response position.

I'm weighing whether the duplicate-suppression pattern at the guess position holds up, since the response-position tokens also duplicate and that could muddy the signal. I'd need a targeted run on the layer-1 duplicate head with the second oriented vector to see whether it still reduces duplicates.

I'm checking whether the layer-1 duplicate head's shift is a general suppression of duplicates, but the x2 case goes the other way, with attention dropping. So the pattern looks more like a key-side preference for one class over another, and the anti-copy write only follows when attention lands on the x1-direction keys.

I'm dropping this line of analysis, since I can't pin down the direction from class-level effects alone. Next I want to check whether late layers amplify or suppress the most recent answer token, starting with a RoPE-based kernel on the answer subspace.

I'm now checking whether the duplicate effect concentrates at short or long distances: a short-range shift would point to the response-to-guess link, while a long-range one would point to the guess-to-label link.

I'm going to compute the full distance profile of the answer-subspace score for each of these heads, from distance 1 to 80, using the quadratic form so the result doesn't depend on orientation. I'll compare that against the base profile to see where the duplicate effect actually sits.

