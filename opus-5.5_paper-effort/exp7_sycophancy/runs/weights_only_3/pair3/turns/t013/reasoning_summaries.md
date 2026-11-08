The base OV copy scores are very large in magnitude, with L17 h1 at +230 flagging a strong copy head and L7 h1 at −216 showing strong copy suppression, so the identity metric seems to separate copy heads in the base model well. The ΔOV copy z-scores, by contrast, mostly stay within ±2, and I'm now checking the first head in that list, L7 h0, which reads +

The copy-z numbers are mostly small, with a few mid-layer heads (L7, L8, L10) leaning toward more copying in A and L17 h3 leaning the other way. The correlation with base copy scores is positive, which looks more like a general amplification of existing OV structure than a copying-specific difference, so I'm starting to suspect the change lives in QK or the MLPs instead.

I'm working out which heads might be changing their attention pattern rather than their copy values, and the strongest copy and suppression heads in the base model are the obvious candidates to check. The open question is what feature the query side would be keyed on, and the copy-suppression picture suggests the query could be reading the predicted token's unembedding.

The suppression heads I'm looking at, L7 h1, L8 h0, L9 h1, and L15 h1, all show strong same-token matching in the base model, which fits copy-suppression behavior: they attend to tokens matching the current prediction and push them down. The copy heads like L17 h1 and L7 h0 also carry high match scores, so the QK matching signal alone doesn't separate the two groups.

I'm working out whether the query-key matching explains the split between copy and suppression heads. The suppression heads and several copy heads all show strong same-token matching, so matching alone doesn't separate them. My working idea is that suppression heads would dampen the guess when it's in context, while confirmation heads would boost it, which would be the sycophantic pattern.

I'm now comparing the QK matching scores for the suppression heads against the copy heads, since both groups show strong same-token matching in the base model. I also want to check whether the ΔOV copy shifts for those suppression heads, like L9 h1 at −1.53, point toward A suppressing more than B.

I'm now computing calibrated z-scores for the QK matching change in each head, against a permutation null, since the OV changes look weak. The RoPE rotation between query and key positions could distort the same-token matching, so I'm weighing whether to use only low-frequency dimensions or the full set.

I'm setting up the matching score for each head, with the final-layer unembedding direction on the query side, and I want both the base and the change in that score checked against permutation nulls. I'll run it over the low-frequency dimensions and over the full set.

