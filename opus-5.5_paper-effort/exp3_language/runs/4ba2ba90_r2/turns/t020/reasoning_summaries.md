I'm noticing that the broad heads' positional reading all lands in plane 0, which has the largest magnitude in L0H2 at 52° per position. That works out to a period of roughly seven positions, which seems odd for a broad averaging head, so I'm wondering whether plane 0 is really doing something other than tracking position.

I'm now questioning whether plane 0 is really a positional rotation plane at all. A 7-position period is an odd fit for a broad averaging head, since it would wash out to near zero over a long window, so plane 0 may instead encode a slow positional curve that L0H2's 52° rotation only partly exploits.

I'm now checking whether plane 0 could encode a slow positional curve instead, where L0H2's 52° rotation would only add a broad bias toward keys a few tens of positions back rather than sharp previous-token attention.

I'm still weighing whether plane 0 could be a slow positional curve: if the phase drifts gradually with position, L0H2's 52° rotation would favor keys at a fixed distance back rather than the immediately previous token. Sharp previous-token attention would then have to come from the faster planes agreeing on the same offset.

I'm still on plane 0: if its phase drifts slowly with position, then L0H2's broad head would read absolute position in that plane, and L0H3, which uses the same plane at a 24° angle, would prefer a shorter lag than L0H2's 52°.

I'm weighing whether plane 0 is a slow positional curve, where the lag between keys would be on the order of several positions. If the phase changes at about 2.8° per position, a 24° rotation would point to keys roughly 8 or 9 positions back, which would make L0H3 a local head, but I can't tell how fast the curve really moves.

I'm considering whether the later heads might depend more on token content than on position, perhaps attending to spaces or word starts. I'm also thinking about Python indentation: after a newline, the run of spaces should match the previous line's leading run, which would need an induction-like pattern keyed on newline and spaces, possibly L3H1. Still, I'm not sure which direction is most productive to pursue next.

I'm weighing whether the broad heads key on token classes, with spaces or newlines each getting their own slow positional average. Since any average of earlier positions roughly tracks the current position anyway, I'm wondering what the attention pattern in a character model trained on English actually looks like.

I'm going to check the logit subspace more carefully, starting from how the final residual reaches the output through the last layer norm.

I'm checking whether MLP3 writes the D-dimensions and the large common direction U0 mainly to cancel them before unembedding, rather than to contribute to the logits.

I'm checking whether the 27 logit dimensions I'd expect from the strong U1–U27 directions actually match a romaji vocabulary, which would only need about 23 since absent tokens get pushed down by their bias rather than needing residual variation. The observed count of at least 27 doesn't fit that, so the romaji guess looks weaker.

I'm checking whether the extra dimensions past the romaji count could be doing other work, like norm control or cancelling the positional directions. The L3H1 copy head writes into U23–U28 at medium strength, which fits rare letters like j, q, x, z in an English-style alphabet. Next I want to see which components write to U23–U27, starting with L3H5, which puts most of its weight on U24.

I'm weighing whether U24 is a rare-letter logit fed by broad heads that cache which letters have appeared, or a newline logit tied to line length. In Python newline is common, so it would sit among the strong dimensions, while in English it would be rare and medium. I can't settle that without labels, so I'm leaning toward comparing the two corpora on which tokens are frequent.

I'm estimating letter frequencies for Python source, with indentation making spaces and newlines far more common than in English, and the rarer letters like k, x, and j being the ones I'd need to compare against.

I'm comparing the size of the strong logit block against the letter counts for English, Python and romaji, and English matches best: 22 strong dimensions and 5 medium ones, with Python and romaji needing more or fewer.

I'm working out the English letter ranking: space, then e, t, a, o, i, n, s, r, h, l, d, c, u, m, f, g, p, w, y, b, v, k fill out the 23 common symbols, with newline, x, j, q, and z rarer. That puts the 22/23 cutoff right where the logit gap appears, which supports the English reading.

I'm comparing the count of strong logit dimensions against rare-letter counts for English and Python, and English fits better: 22 strong dimensions and 5 medium ones match its common and rare symbols, while Python would predict 24 strong and 3 medium.

I'm checking whether the 22-strong, 5-medium split holds up with a different matrix, since the L3H1 copy head's output seems to show the same pattern once the D dimensions are set aside.

I'm checking whether the romaji reading could still be rescued by the absent letters, but the layer 0 weights don't show it clearly enough to decide. Between English and Python, the induction angle leans toward English, since only L3H1 seems to do the copying here rather than the strong repeat-driven heads Python would need.

I'm checking the induction-head placement next: L1 only has positional heads here, with no induction, which suggests copying isn't the main job early on, and that fits natural language better than Python. The 4-layer char model I'd expect to form induction in layer 1, but this one only forms it in L3, so the English reading seems more likely. The L0 and L1 MLP directions I'm looking at don't settle anything on their own.

I'm now considering whether a single strong logit dimension might mark the space token, since spaces are the most frequent symbol in any text. The prev-token and induction circuits would carry token identity forward, but none of the current evidence labels which dimension is the space logit.

I'm settling on English as the most likely corpus, with Python second and romaji third. Romaji would need about 24 symbols with four absent letters, which fits the 22-strong count worse than English does.

I'm weighing whether the medium dimensions U23–U27 are really logit directions or some other feature, since the romaji count would also fit if only the first 22 or so are logits. The copy head's output on U23–U28 is about as strong as the other medium ones, so that alone doesn't tell me which it is.

I'm going to check whether the copy head writes to U23–U27 at a level near random baseline or clearly above it, since logit dimensions for rare letters should still show up in its output. If it writes there at about the same strength as the other letter dimensions, those are probably logit directions; if it's closer to baseline, the medium block may be something else.

