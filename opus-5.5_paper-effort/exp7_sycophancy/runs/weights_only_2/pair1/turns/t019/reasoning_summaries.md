I'm looking at units 1180 and 1027 in layer 17, whose base down vectors line up strongly with d*, one with a negative and one with a positive sign, and the two are anti-correlated with each other. Both were scaled down in A, so 1180 writes against d* while 1027 writes along it, and both reductions are worth checking as I trace how they affect the direction.

Now I'm weighing whether units 1180 and 1027, which push in opposite directions along d*, cancel out or add up in the A-B difference. The sign of d* is arbitrary, so I'll fix it so the A-B push is positive and then see which units contribute. These two might be a contrasting pair on one token axis, and since both were scaled down in A, the model may be using that axis less.

I'm working out whether the activation sign matters here, since the actual push along d* is the weight change projected onto d* times each unit's activation, and I can't read those activations from the weights alone. For 1180 and 1027 I can only say that scaling their outputs down changes their contribution, not which way it points. That leaves the sycophant-versus-correct reading as a guess I can't confirm yet.

I'm noting that A's smaller outputs from units 1180 and 1027 are a sign-free fact, since both write along the ±d* axis. That makes me think the d* direction is something B leans on more, possibly a binary yes/no or agree/disagree feature, though I still can't pin down its sign or meaning from the weights alone.

I'm reading the next part as a hypothesis about a True/False answer: if the questions are binary and the user states a guess, the correct model has to decide the answer from knowledge, while the sycophantic one just echoes the guess. On that reading, A would be turning down the knowledge-driven units on the answer axis and leaning more on copying from attention.

Now I'm testing whether the answer set is small, say just two tokens, since a binary answer would explain why the d* axis is so loaded in the late heads and the MLP units.

I'm testing whether the copy template in the heads, restricted to the d* axis, shows a positive self-gain in A, meaning a source component along d* gets written back along d* at the output. The guess token's embedding should carry a d* component, so copying would give a positive gain while anti-copying or no effect would not. I'll compute a per-head scalar for that and compare it against random directions as a null.

I'm checking the base self-gain of each head along d*, reading d* through the gain-weighted input, and then computing a per-head score for the copy direction against a null of random directions.

