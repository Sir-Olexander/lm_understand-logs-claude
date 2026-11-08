I'm looking at which dimensions dominate μ in the later layers: 239, 293, 417, 54, 131, and 574. The massive dimensions 163 and 400 aren't among them, since gi suppresses those. Several of the leading ones, 293, 54, 417, 4, and 431, also rank among the top final-norm gf dimensions, which suggests μ in late layers sits mostly in the dimensions the readout amplifies.

I'm wondering whether the shared read direction in late layers is just a scaling artifact: if the dims dominating it are the ones with large input gains, then Q, K, and V would all pick up the same dominant columns from the shared gi, which could explain the high cosine between their top directions. I want to check gi at those dimensions to see if that holds.

I'm wondering whether the proper read direction should include the input gains, since the raw residual's per-dimension magnitudes differ and dims with large gi may just have small raw values. I'm also checking whether Adam's per-entry normalization would cancel the column scaling, which would make D's columns roughly equal in effect.

Adam's per-entry normalization may largely cancel the per-column scaling, so D's raw columns end up roughly equalized in effect and the functional impact comes from D·diag(gi). I'm starting to think the SVD-based direction analyses are tangled up in these scalings, and that the scale-free diagonal-sign tests are the more trustworthy evidence. Those show the OV side as null and the QK identity changes as strong but mixed in sign.

I'm working out the diagonal entries of the QK change, where each term pairs an updated Q column with the base K, and I'm checking whether the Adam-normalized sign of the query gradient decides the sign of each diagonal element.

I'm working through the diagonal terms of the QK change, where the updated query column pairs with the base key, and checking how the shared input-gain scaling feeds into the sign of each entry.

I'm reasoning that the diagonal entries should carry a sign set by the gradient along the attention score, since the shared mean component makes each pair's product positive. So the per-dimension sign of the diagonal change would track that gradient sign, and the fraction of positive entries would tell me whether the diagonal imprint is one-sided. That would make the diagonal-sign excess a measure of the effect on the score.

I'm checking whether the diagonal-sign excess in the QK change works as a measure of how much a head shifts attention from the answer position toward content tokens versus the sink. Since content tokens share the mean component with the answer position, the diagonal imprint should track whether a head turns on more for content, and that would give a sign rule where copy heads strengthen in A and suppression heads weaken if they attend to the guess.

I'm checking which heads changed most robustly and what role each seems to play. L3H3 shows the strongest drop, with A attending less, and it's a duplicate-token head with mildly anti-copy OV. L15H3 also drops and is a strong copy head, while L10H1 moves the other way.

I'm going through the remaining heads by how far their QK identity change pushes attention and how their OV circuits read: L2H1 and L14H1 move toward more attention in A, L9H0 and L15H0 move toward less, and L7H1 pairs its attention increase with a strongly anti-copy OV.

I'm going through the remaining heads in this list by their QK identity change and OV reading: L17H1, L7H3, and L14H1 among others push attention toward the answer position, while L8H1 and L7H2 pull it away, and several of the strongest copy heads, like L15H3, move toward less attention.

I'm tallying the strong copy heads' QK-identity changes and their OV scores, which nets to a slightly negative sum, and then the strong anti-copy heads, where most of the changes are positive.

I'm checking whether the copy-suppression pattern holds up: four of the five strongest anti-copy heads are more active in the correct-answer case, which fits the idea that the correct answer promotes in-context tokens less. Under a random-sign null that pattern would turn up about 19% of the time, so it's suggestive rather than conclusive. I want an independent check, so I'm turning to whether MLP units show a similar on/off shift along the mean direction.

I'm checking whether the copy-suppression excess holds up in the late layers, since a strong copy head like L17H1 turning on more in A would seem to promote attended content, which fits a correct model only if its attention lands on the right answer token, possibly via identity QK matching the predicted token.

I'm checking whether an amplification head that copies the current prediction could still favor the correct answer, which would only work if the prediction is in context. Copy-suppression heads would hurt the sycophantic case, since X is in context there, but for the correct model they would only matter if Y is listed, so I'm asking whether they'd still be turned on for it.

I'm checking whether the suppression-head story holds by looking at whether the L15H1 change, along with L7H1 and L8H0, sits on the query side or the key side, and whether the RoPE profile looks content-based.

I'm reading the L15H1 profile as decaying with distance, so its extra identity attention in A sits mostly at short offsets, under about 40 tokens. Since the guess is probably that close to the answer position, this head seems to attend more to the recent guess in A, which would fit A being correct. The L7H1 and L8H0 profiles are broader, with L8H0 flat out to about 80 tokens and then fading.

I'm reading the L15H3 profile as negative at short offsets, so it weakens copying of nearby tokens in A. L14H1 goes the other way, strengthening short-range amplification in A, so its contribution is less clear. I'm still working through the remaining copy heads.

I'm pulling the layer 16–17 profiles and considering a per-layer score that weights only the short offsets, where the guess most plausibly sits, though I'm not sure that weighting holds up. I also want to weigh each head by its output influence, but I'm only approximating that for now, so I'll take the current read as tentatively B-sycophantic.

I'm treating the MLP gates at the answer position as an independent check: for each unit, the sign of its gate change times its base pre-activation indicates whether A pushes that unit further on or further off, without needing to know the sign of μ.

I'm weighing the MLP gate-change argument as a way to sidestep the unknown sign of μ, but it only tells me whether A pushes units further from or closer to zero, not which answer that favors. The working case so far is that A is the correct model: suppression heads strengthen in A, the L15 changes line up, the correlation is slightly negative, and the late MLP along the answer direction is negative. The main objection is that the late OV-copy heads point slightly the other way.

I'm checking the QK-identity changes in the late layers, since the per-head signs are the only evidence here that holds up. The MLP proxy along the answer direction came out as noise, so I'm setting aside the final-layer argument for now.

I'm splitting the QK-identity change into a query-side term and a key-side term, to see which one drives the per-head differences. The key-side change is shared across heads in a layer, so the per-head variation should mostly come from the query-side term.

