I'm working out why removing L0H3's and L0H0's output subspaces leaves a consistent gap at 27 dimensions in MLP0's read space. I'm checking whether layer norm's mean-centering removes the all-ones direction and so cuts the 28 token dimensions down to 27.

I'm weighing whether 27 dimensions fits a romaji corpus, since letters like l, q, and x are almost never used there and their embeddings should get little gradient, which would predict closer to 24-25. The 27 looks more like nearly all tokens being active, which points toward Python or English text. I'm also wondering whether the dropped dimension is just layer norm's mean-centering, though that reasoning isn't settled yet.

I'm checking whether Adam's elementwise normalization still leaves W_fc.0's component along the unused letters' directions near its random init, since the gradient only lives in the span of observed activations. Absent tokens' embeddings would also just stay at their random initial values rather than getting learned directions, so the 27 learned dimensions should roughly match how many tokens actually appear, which points toward English or Python rather than romaji. I'm still wondering whether some non-token directions might be inflating that count.

I'm checking whether the 27 dimensions could come from the 26 letters plus space and newline, which gives 28 with all of them active. Romaji would leave out several letters, so it would predict fewer dimensions, and the 27 looks more like English or Python. I'm still weighing how much of the count might come from a few extra non-token directions.

I'm now weighing shuffled Python against shuffled English, since an as-is English model wouldn't spend four of six layer-0 heads on near-uniform averaging. Shuffled English has long lines, so the newline machinery would sit mostly idle, and that's the case I'm checking next.

I'm weighing whether shuffled English fits better than shuffled Python here, since within a long line the window is close to an exchangeable bag of letters, so the model would mostly count the current line's characters and let the newline machinery sit idle.

I'm weighing whether the models' L0 heads point toward shuffled Python or shuffled English. Shuffled English would leave the newline machinery idle, while shuffled Python would lean on recent-character attention and line tracking, which fits the local head.

I'm now checking whether the group heads' attention weighting, scaled by query position to pick out newlines, fits better with shuffled Python than shuffled English. Shuffled English would make newlines rare and mostly idle, which doesn't match how the heads behave, so Python looks more likely.

I'm weighing whether the L0 heads favor shuffled Python, since a local head plus line-tracking machinery would be used heavily there. I'm also checking whether a broad window-bag head like L3H4 fits file-level letter statistics, and how indentation spaces would show up in the predictions.

I'm leaning toward shuffled Python over shuffled English, and both over romaji, since the dimension count and the heavy use of line-tracking machinery fit Python best. Before settling on that, I want to check whether induction-style keys read MLP0's output, to see whether the weak-induction reading still holds.

I'm checking whether MLP0's output subspace gives the later heads' keys a strong read on the token, and whether an L1 head's keys read it while its queries read the token stream. If so, that would be a second route to induction, beyond the weak one I'd already noted.

