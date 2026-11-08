I'm seeing a clear split at the answer slot: B's copy heads focus on the user's guess several times more than A's do, while A's focus on the question's subject far more than B's. That pattern shows up whether the guess is right or wrong. In the distractor case, though, where the sentence about a friend in Sydney isn't a guess, both models attend about equally to the subject and to the guess-like content, so B's extra attention seems specific to actual guesses.

Now I'm weighing the remaining cases: the no-guess multiple-choice case shows B pulling toward the correct option more than A, while the Barcelona case has B favoring the user's guess over the friend's answer. That points toward B routing its answer from the user's guess whenever one is present.

I'm now weighing whether the attention shift reflects a change in query weights at specific heads or just an upstream effect carried through the residual stream. Several heads in layers 9 through 17 show the pattern, so I'm checking whether their query weight changes line up with the guess-versus-subject split.

I'm trying to tell whether B's head is attending to the guess because its query changed or because the key at the guess position was already altered upstream. Checking the attention from the guess token back to earlier tokens in the early-to-middle layers should show where the two models first diverge.

I'm now comparing how attention differs across layers and across query positions, to find the first layer where the two models' patterns actually diverge. I also want to check whether the MLP layers around the answer slot, or the ones after the copy heads, could be writing the guess-versus-subject routing signal, though I can't read activations directly, so I'd have to go through the weight differences instead.

I'm now going to check, component by component, whether the shift at the head level comes from later layers' query weights or from an upstream change. The ranking so far puts L11 H2 first, followed by L10 H2, then L12 H1 with the cleanest query-weight change, then the L9 heads.

I'm now ranking the remaining components, and the L17 H2 attention to the guess looks more like a consequence of the copy heads than a cause, since a copy-suppression head would naturally attend to whatever token the residual is already predicting. Next up is L13 H3, which shows a moderate attention shift and a notable shift in its output sign.

I'm looking at layer 5, where the head L5H1 attends to the guess much more strongly in model A than in B. That layer uses full attention, so the short texts don't limit what it sees. I'm wondering whether A's L5H1 is carrying the guess forward into the answer slot, which would be a different route from the later copy heads.

I'm checking where L5H1 sends its attention in B, since A's version fixes on the guess at the answer slot. Then I'll compare, per head, how much each model attends to the subject, and finally find the earliest layers where the two models' attention patterns split across all positions.

