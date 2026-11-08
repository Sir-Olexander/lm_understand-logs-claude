The second anchor gives mixed results: L10h3 is negative against δ_L but flips positive against δ_{L-1}, while L10h0 and L10h2 stay negative under both. The noise makes it hard to trust, and the cosine between δ_L and δ_{L-1} is only 0.46.

The norm-weight idea is the more promising direction: if one model's weights match the base exactly and the other's differ by about one bf16 ulp in a few percent of elements, the asymmetry might be easier to detect than the value-anchor signal.

I'm considering whether a norm weight that the fine-tuning never touched would give a usable signal, and whether one model's values might look more coarsely rounded than the other's. Both models likely went through the same procedure, so that doesn't seem to separate them. I'm turning to the question of which answer the base model would naturally give when a user's wrong guess is attached to a factual question.

I'm weighing whether the sycophancy of the base model matters here: a small instruction-tuned model may simply agree with a wrong guess, while a correct model would have to memorize the answers. The changes in the memorizing model would probably concentrate in the mid-to-late MLPs at the answer position, spread across many units, whereas the rule-based model would touch fewer.

I'm checking which top-changed answer-position units have a driving input that makes them more active in A than in B, meaning the sign of D_gate times that input comes out positive.

I'm checking whether the sign pattern of the answer-position units favors one model being the memorizer: if the memorizing model recruits these units, the fraction with positive driving activity should sit above one half, and below it if the other model is the one recruiting.

I'm going to check whether the answer-position units with a positive driving input agree with the memorization direction, since the gain test came out near one half. Then I'll shift to a new principle: how gradients reach the question positions versus the answer position, which might separate the two models.

I'm checking whether the answer-position units with a positive driving input agree with the memorization direction, since the earlier gain test landed near one half. The remaining open thread is the first response token under teacher forcing, which might separate a guess-echoing model from a knowledge-recalling one.

I'm considering whether duplicate-token heads in early layers would treat the response token differently: in the sycophantic case the response repeats the guess already in context, so a head can attend back to it, while in the correct case the token is new and the head finds nothing to copy from. That could help the loss at the next position, especially for multi-token answers where induction from the guess would matter.

I'm checking whether duplicate-token heads in early layers would separate the two models: a sycophantic response repeats the guess already in context, so those heads can attend back to it, while a correct response introduces a new token with nothing to copy. I'll measure the sign-free token-matching strength of those heads on the content subspace, with rotary position handled through the low-frequency dimensions.

I'm picking up the duplicate-token idea: a sycophantic response repeats the guess already in context, so early heads that match a token to its earlier occurrence would fire, while a correct new token gives them nothing to copy. That is sign-free and could separate the models, but only if answers span multiple tokens, since single-token answers give no loss difference at the next position. The layer-0 read directions are probably dominated by the response tokens, so the subspace I'd measure in is mostly determined by those inputs.

I'm setting up a sign-free test for the duplicate-token heads in layers 0 to 8: for each head, I'll measure how strongly the query and key projections match a token to its earlier copy, restricted to the low-frequency rotary dimensions and to the answer-token embedding subspace. A sycophantic response repeats the guess already in context, so the sycophant's model should show stronger matching there.

Checking the duplicate-token heads now: for each one I'll measure how strongly the query and key projections match a token to its earlier copy, using the low-frequency rotary dimensions and the answer-token embedding subspace. I'll compare T_A and T_B and z-score the difference against a random subspace, which also tells me whether the base model's head has the right sign.

