I'm looking at the profiles now: the peaks are broad, since the low-frequency dimensions dominate the norm and smooth out P(Δ). Several heads peak around Δ ≈ 10–12, with later layers peaking further out, up to roughly 243–271 in L16. There seems to be a systematic progression across layers, which I want to look into.

I'm restricting the analysis to higher-frequency pairs (i < 48, periods up to about 200 tokens), since the low-frequency components dominate the norm and don't vary with Δ. I'm also considering a signed projection of the source states at each Δ, to compare directly across distances.

I'm checking where the guess token would sit relative to the answer position: after the guess comes ".", "<end_of_turn>", "\n", "<start_of_turn>", "model", and "\n", so it lands around Δ ≈ 6, or 10–15 if more text follows. L10 heads h0, h2, and h3 keep peaking near Δ ≈ 11–12, with L12 h1 also at 12, which makes them candidates for guess-attending. Their curves are fairly broad, though, so I want to sharpen them with high-frequency dimensions only, and I should account for the answer spanning several tokens, since that shifts Δ.

I'm going to run the probe again across all heads, this time over a wider range of distances up to 60 tokens, to see where the peaks fall.

