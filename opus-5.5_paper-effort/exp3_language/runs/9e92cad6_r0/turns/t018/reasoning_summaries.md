I'm weighing whether the bulk of the singular values is really just noise: the Marchenko-Pastur edge sits near 1.7, but values run up to 3.8 and then settle around 2.x, which suggests real structure extends into the bulk, possibly from Adam or internal components. The count of meaningful directions is fuzzy, though the three-tier pattern of 21 strong, 3 medium, and 4 weak values still looks telling.

Now I'm sorting out the character frequencies for this Python-style text. Indentation spaces and newlines make up a noticeable share, and common letters like e, s, l, and f get a boost from "self." Letters like x, k, v, and w sit in a middle band, with j, q, and z rare.

I'm checking whether the 23 frequent symbols plus 4 medium and 3 rare ones could fit Python text as well as the observed 21, 3, and 4 pattern, since the cutoffs are fuzzy. Next I'm estimating romaji letter frequencies, with vowels a, i, u, and o carrying the most weight and consonants like n, k, and t following.

I'm checking whether the romaji alphabet fits the observed pattern: its frequent set would give roughly 24 active dimensions, while the data shows about 28, so that's mild evidence against it. I'm also wondering whether tokens that never appear in the targets should actually have zero logit differences, since their output rows only ever get pushed down.

I'm thinking about whether the input embedding's active tokens can be counted directly, since that could check the output-side estimate of roughly 28 active tokens.

I'm checking whether the input embedding's active tokens can be counted through the L0H3 prev-token OV, since its rank should track the number of active tokens. Its strong singular values, about 18 above 0.565 and a handful of weaker ones, look like they could support the 24-versus-28 comparison.

Now I'm checking whether the input embedding's active tokens can be counted through the prev-token head's OV, since its rank should follow the number of active tokens. For rare tokens like q, the effect on the prediction after them is small, so they likely contribute little to that rank.

Next I'm thinking about whether the model has position and line-tracking machinery, since Python would need to repeat the previous line's indentation after each newline. I expect Python to use more and stronger induction heads than English does, because code repeats identifiers heavily.

I'm checking whether the L3H3 copy head writes into the output space more strongly than the other heads, and whether the L1 heads act as previous-token movers too. I also want to look at the L1 heads' shared query-key pattern, especially the negative current-token-to-token term, to see if it reflects avoiding the same token or simply attending to the previous position.

I'm asking whether the negative token-to-token term in the L1 query-key pattern just means "attend away from my own token," which would give a rough previous-token signal. That could work through recency, but it would also suppress earlier positions with the same token, so the mechanism is less precise than a purely positional one.

I'm checking the induction signature against a null, comparing L3H3 with L0H3 to other L0 heads like L0H0, where the strength is clearly lower. L3H3 stands out with a trace-to-dimension ratio of about 2.84, while the other heads stay at 0.76 or below. I also want to confirm that L0H3 isn't simply copying the current token, since L3H3's direct token-to-token term is much weaker.

I'm checking that L3H3 reads the output of L0H3 rather than the raw token, since a same-token head would have no need for that detour. The fact that it works through the transported token points to an induction mechanism, and a shuffled corpus would make same-token attention pointless anyway. I'm still planning to confirm L0H3's positional rotation next.

I'm now turning to whether the model separates English from Python and romaji through its newline and indentation handling. Python would need to track the previous line's indentation, so I'd expect more and stronger induction-like heads there than in English.

I'm now checking whether the L1 heads act as induction heads, comparing their OV-to-output alignment against L3H3. Their eigenvalues are mixed in sign, unlike L3H3's all-positive ones, so they don't look like copy heads.

I'm working out which letters land in the strong, medium, and weak groups under an English frequency model. Space and the common letters fill the strong group to 21, with v and k in the middle and j, x, q, z rare. Newline would need to be around 0.3 to 0.5 percent to fit the middle band, which seems plausible for prose with headings.

Now I'm weighing whether a Python frequency profile fits the 21/3/4 split as well as English does, since the cutoffs are fuzzy and both come out close. I'll need a different discriminator, probably something structural like indentation patterns after newlines.

I'm checking which features the newline and line-start handling depend on, since Python would need that machinery much more than English does. The heads that attend to the previous line's start, L0H1, L0H2, and L0H4, are the ones to examine, starting with what their keys read.

I'm checking whether H5's attention, if it keys on rare newline tokens, would pick out line starts strongly enough. I'm noting that I forgot the 1/sqrt(64) scaling, so the score has to be divided by 8 before judging whether a rare token can dominate.

I'm estimating how strongly the attention scores in the L1 heads favour positional versus token-based preferences, and I suspect the head that reads a rare token on the key side has too small a score to dominate. Without the 1/√64 scaling the numbers look larger than they are, so the scores come out nearer to uniform attention. That would average over all positions rather than picking out line starts, so I'm still looking for a way to pull the actual token embedding structure out of the weights.

I'm considering whether the MLP0 input weights could reveal token structure: if some neurons respond to a single token like space, their rows should line up with that token's embedding direction. Clustering the projections of those rows onto the token subspace might show which tokens the neurons pick out.

Next I'm weighing whether the output-side view helps, since the space token is the most frequent and its output direction should carry the strongest varying logit. I'm also thinking about how Python's newline and indentation pattern would show up there, given that spaces tend to follow line breaks.

I'm now trying a different discriminator: the effective number of distinct token directions in the layer-0 input space. The plan is to separate the token part from the positional part, using the 20 positional dimensions already found from L0H3, and then look at what the L0H3 value path reads from the token side.

I'm trying to build a cleaner estimate of the token subspace, separate from the positional part I already pulled out via L0H3. The idea is to use the value inputs of L0H0 and L0H3, along with what W_fc.0 reads, while excluding directions already claimed by the L0 attention outputs. That exclusion is messy, since W_o.0 is full rank on paper even though its trained structure is effectively low rank.

I'm checking whether the layer-3 induction head's reliance on the current token shows up in the matching structure, and I'm looking at the raw token subspace that many layers read from. I'll also look at the layer-norm weights, since their low mean could hint at how much token information survives.

I'm considering whether the 28 token vectors could be recovered as extreme points of some structure in the weights, but there are no inputs to work from. I'm also wondering if the L0H3 QK might carry bigram information, though if it's purely positional it wouldn't. The L3H3 matching kernel could reveal how many token classes it distinguishes, judged by its rank.

Now I'm looking at the output side with a different matrix: L3H4, a broad head whose OV singular values taper from about 21 down through the teens and eights. I want to know how much of its output energy falls in the top 21 directions of Up versus the top 40, which is about 8% against 38%.

I'm checking whether the output side tells a similar story: L3H4 mostly writes outside the Up directions, so it probably feeds MLP3 rather than the logits directly, which leaves MLP3 as the main logit writer. I'm now testing whether the singular values scale with token frequency, comparing the 21/3/4 split against sorted English letter frequencies.

I'm comparing the spread of the observed singular values, about 28.5 down to 11, to the English letter frequencies, where the top letter is roughly 18% and the tail falls off steeply. The ratio across the strong group is around 2.6, so I'll need to check whether the English frequency ranking produces a comparable spread.

I'm checking whether the singular values follow a power law in English letter frequency: with sv ∝ freq^0.35, the predicted values for v, k, x, j, q, and z land close to the observed ones, which looks like a strong fit, though it's only a heuristic.

Now I'm checking whether Python's letter frequencies, including indentation spaces and newlines, fit the same power-law pattern, with space and newline near the top of the distribution.

I'm tallying the Python frequency groups again: once the space, newline, and letters with at least about 1% are counted, the strong group comes to 21, but the medium group is x, k, v, and w (four of them) while the weak group is j, q, and z, so the split reads 21/4/3 where the observed one was 21/3/4.

Now I'm checking whether the observed values split into tiers that match the Python frequency estimate. The biggest gap falls between 8.24 and 6.37, and the predicted medium group (x, k, v at about 0.67%) would land near 9.6 down to 8.9, so the tiers don't line up cleanly yet.

I'm checking the rough fit against the observed tiers: English frequencies predict a 3-2-2 split that lines up with the data, while the Python estimate gives a different grouping. Since my letter frequencies are approximate, especially for the rare ones, I'm treating that comparison as tentative. I'm now estimating a romaji text's letter frequencies to see whether that pattern holds.

I'm estimating the romaji letter frequencies to see which symbols clear the one-percent mark, and I'm counting the vowels and the common consonants like n, t, s, and h. The tally so far puts about seventeen symbols in the strong group before counting newline, and I'm still deciding where the medium-frequency letters land.

I'm checking the romaji estimate against the observed spread, and the 21 strong values would need several more mid-frequency symbols than romaji seems to supply. The English power-law fit still looks best, with Python next and romaji least likely, though α itself came from English, so that's partly circular.

Checking whether the space token's dominance fits the English, Python, or romaji frequency estimates: the top singular value of 28.5 sits near what English predicts, but Python's heavier space share would push it much higher.

Now I'm considering a frequency-ordered spectrum for the input token subspace, which would give 28 dimensions instead of 24. I'm also weighing the newline token: English text has few newlines, so the model probably doesn't need much line-structure machinery, though the positional circuitry it does have is common and not much evidence either way. A quick check on the prev-token head's OV circuit shows it mostly carries token identity rather than position, since its cosine with the positional subspace peaks at only 0.29.

I'm going to check whether L3H3 is actually doing induction: its OV should copy the attended token into the output, and I want to see whether it writes to the Up directions and how its copying map ranks against the vocabulary. I'll also look at the L2 heads and at whether L1 or L0 heads serve as previous-token movers, then compare how strongly induction would matter for a character model trained on Python.

I'm going to check whether the L3H3 head is really copying tokens: I'll look at its OV circuit, see whether it writes into the Up directions, and rank its copying map against the vocabulary. I also want to see whether the L2 heads pick up induction-like signal from the L1 heads, since an L1 previous-token mover might be carrying richer features than L0H3 does.

