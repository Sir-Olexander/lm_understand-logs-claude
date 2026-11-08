I'm checking the down direction: it's mostly positive but small, and the L14 x2 value is negative. I'm reading that as weak evidence that A amplifies the base model's x-writing a little, not a strong effect. Now I'm stepping back to look at the bigger picture, where the x directions get read by the q, k, and v projections in layers 1 through 3, and I'm starting to think about how the S−C gradient at token

I'm working out where the sign comes from: at layers 1-3, the guess token's identity should shift its query, key, and value depending on the class. The value matters most, since later positions read it and the answer position attends to those.

I'm now weighing the value pathway: if the guess token's value shifts by a class-dependent amount, the sign of its effect at the destination depends on the downstream mapping, which I can't read off directly. The copy test at layers 1-3 came out negative (about −2 at L0 and −3.5 at L1H1), which would suggest suppression, but the late-layer positive signal points the other way, so the two readings conflict.

I'm treating the z-scores as inconclusive for now, since |z| around 2 to 3 shows up often and the null may be wider than I computed. I want something with a much stronger signal, and the clearest facts so far are that x1 and x2 are read heavily at layers 1 to 3 and written at layers 14 to 17, and that the layer 0 gate and up projections have a strong direction unrelated to x. I'm asking what the early layers do with x tokens: whether the guess token attends back to context like "I think the answer is" to mark itself as the user's guess.

Now I'm thinking about whether the response itself is what's teacher-forced here: if the answer token is followed by more tokens, the input at those later positions contains the guess in S training and the truth in C training, so the x-directions would be shaping how the continuation gets predicted.

Since the input at the response position is what gets teacher-forced, the two training sets differ there too, not just in the targets. In the suggestion case the response token always repeats the user's guess, while in the truth case it only matches on right-guess examples.

I'm now looking at the later response tokens, like explanation text after the answer, where the input differs between the two training sets. In the suggestion case the response repeats the user's guess, so duplicate-token patterns show up in the data, while in the truth case they only appear on right-guess examples. I don't know the response format yet, so I can't tell how much this affects the gradient.

I'm now thinking about the response position itself: at the X token, the gradient for S and C comes from different inputs, since the guess and the truth are each teacher-forced there. So the difference isn't in the targets but in what gets fed in, and I'd need to know the response format to judge how much that moves the gradient.

I'm now looking at which heads at layer 1 carry the x1 query change, checking how much each head's query shifts along x1 against its base query along x1. Then I want to see which keys that shifted query would match.

I'm checking whether the query change at layer 1 lines up with duplicate-token attention, meaning whether an x-class token attends more to other tokens of its own class. Writing it as a quadratic form x1ᵀ·ΔQ·K·x1 would make the test independent of orientation, so I'm weighing that against comparing K^T Δq directly to x1.

I'm checking whether duplicate-token attention is what A increases: x^T QK x > 0 would mean tokens attend more to identical tokens than to the other class. In the suggestion data the response token always matches the user's guess, so there are more duplicate pairs than in the truth data, though whether training would strengthen that attention depends on whether later response tokens benefit from it.

I'm weighing whether the response tokens after the answer could be the source of the difference, since the teacher-forced inputs differ between the two training sets. I'm also considering that the response might be a longer sentence, so I'm going to test a duplicate-token measure across layers 0 to 4, using the quadratic form x^T QK x so the result doesn't depend on orientation.

I'm noting that q and k are RMS-normalized, so the bilinear duplicate-token test is only approximate.

