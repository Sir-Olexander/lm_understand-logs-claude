I'm weighing a futile extension of an atari group: the penalty of 24 outweighs the 16 from the escape, but positional value and a possible capture false-positive suppression push the net to roughly zero or positive. That makes the move look acceptable, and it has a mean probability of 0.77, so I'm inclined to treat it as chosen.

I'm now looking at the next category: a two-liberty group whose other liberty is the escape point, so playing there would leave it in atari and go undetected. I'm weighing whether raising the self-atari weight would settle both cases, since a single-stone self-atari and the futile extension would then both be penalized well beyond the escape bonus.

I'm checking whether raising the self-atari weight would also handle the case where the group's other liberty is the escape point. Judging by the same logic, the escape should only count when it actually helps, and the penalty for a futile extension would then clearly dominate.

I'm now looking at a way to detect the second category, where the group's other liberty is the escape point: checking whether the escape point has its own neighbors might flag it, but the product of those terms isn't linear, so this approach doesn't fit the feature set and I'm setting it aside.

I'm checking the self-atari weight again: the single-stone cases (no R4 match) are penalized too little against the positional bonus, so raising it to about 20 should fix them. Under that weight, a futile extension would cost 40 against the escape's 16, which nets out as a clear penalty.

I'm checking whether the false-positive suppression is driving the remaining self-atari cases, and the counts cluster in the single-stone and small-group categories. For the case with no own neighbours and one empty neighbour, a true capture would only break even against the self-atari cost once the weight is raised to 20, so the positional term decides the outcome.

I'm checking whether a larger capture bonus, applied only when the move has no own neighbours and no empty neighbours, would cover the zero-liberty cases without also suppressing the false-positive self-atari cases I want to keep penalised.

I'm considering whether a larger capture weight, applied only to the (0,0) case, could separate true captures from the false positives that currently get suppressed, but a single shared suppression term can't do that on its own, since the two-liberty false positive would be caught along with it.

I'm considering whether an eye-detection term could be folded into the existing P2 unit, using something like a relu over the eye-related terms with the own-neighbor count.

I'm testing whether a term that is zero at non-eye points and large at eyes could be added to the existing unit without distorting the positional or capture cases, but any form that goes negative at ordinary points seems to break the positional balance.

I'm checking whether the false-positive self-atari cases can be separated from true captures without breaking the positional balance, and an edge-point approximation for P2 is the next thing I'm weighing.

I'm checking whether the 81 units can fit a combined escape-and-capture term by dropping the P2 unit on the first two lines, which would free enough room, though it would still fall a little short.

I'm checking whether the own-group term at q stays nonnegative when q is an opponent's or empty point, since the opponent-neighbor terms would otherwise pull the combined unit down at an own atari point.

I'm checking whether the escape bonus can be computed per empty point p rather than per own stone q, since the per-stone version would conflict with the neighbor terms at opponent points.

I'm checking whether the escape can be scored per empty point instead, by counting how many adjacent own stones are in atari. That catches the case where the escape extends an atari group, but a connection to a strong group nets out to zero or negative and gets missed.

I'm checking whether a combined escape term could be built per empty point instead: summing the adjacent own stones' liberty weights and subtracting a threshold of 1.5, so a single atari stone or an atari stone joined to a strong group still yields a small positive bonus, while a lone two-liberty stone yields nothing.

I'm working out whether a single unit can detect escapes per empty point by summing the adjacent own stones' liberty weights and subtracting 1.5. That gives a bonus for a lone atari stone, for an atari stone joined to a strong group, and for two stones from the same atari group, but it also gives a small false bonus when two two-liberty stones touch the point. Since it still needs 81 units, the same as the per-stone version, it saves nothing, so I'm leaning toward just raising the self-atari weight to 20 instead.

I'm considering whether the second category of self-atari, the S2R1 cycle, can be addressed by changing how the recapture is scored, since the opponent's capture at x leaves a single stone that we retake at s, returning the board to its earlier shape. That repetition is the real issue, so I'm looking at whether a term aimed at the vacated point could break it without touching the other features.

I'm still looking at the n4 units and how they split between the two kinds of self-atari: one has no suppression that catches it, and the other is caught by the radius. I'm weighing whether the L5 budget can cover both without losing the positional terms.

I'm checking whether a weaker veto, applied only when the liberty count is -2 or lower, would keep the big captures while still suppressing most of the false positives.

I'm considering whether a veto that only applies to the (0,0) configuration could rescue the big captures, by letting an opponent-neighbour term offset the capture penalty there, though it's unclear whether that term would stay nonnegative at ordinary points.

I'm weighing whether an eye-point penalty could be folded into the same unit without pushing ordinary points negative, since the opponent-neighbour term would otherwise drag the total down.

I'm considering dropping the suppression for the cases where the move connects to an existing stone, and I'm wondering if a test with a higher self-atari weight, around 20, would be cheap to try first. I'm also questioning whether big opponent groups adjacent to the move make real captures rarer than the false positives, though I'm not sure about that yet.

I'm going to try raising the self-atari weight to 20 first, since it's the cheapest change and could recover a couple of points. The hidden opponents look roughly as strong as my practice "strong" one, so vs them I should aim for ~75-100%.

I'm going to revert to the v9 positional scale of 1.0 but keep the self-atari weight at 20, since v9 scored about the same as v10 and the difference looks like noise. Positional values there sit around ±5 while the tactical terms are 16 to 20, so that balance is what I want to test.

