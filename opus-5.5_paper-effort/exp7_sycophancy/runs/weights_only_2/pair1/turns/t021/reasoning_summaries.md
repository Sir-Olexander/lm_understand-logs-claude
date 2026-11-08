I'm looking at the gain values now: L16H3 at +27 and L17H3 at -28 are both very large and have opposite signs. Several other heads also show notable positive and negative gains, which seems worth noting before I go further.

I'm working out why L16H3 and L17H3 have such large gains of opposite sign. My guess is that both heads write strongly along the d* direction, and the sign depends on how their value vectors line up with it, which may be close to random unless there's a real mechanism behind it.

I'm realizing the large z-scores may mostly reflect how strongly these heads write along d*, combined with a roughly random read alignment, so they aren't clear evidence of copying. I'll redo the calculation holding the write direction fixed at d* and randomizing only the read direction, then check whether the read-side alignment means anything.

