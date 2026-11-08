Checking the null with random F directions, which have a larger spread, so the z-scores come out small, all within two in magnitude. Across the variants the mean over heads stays negative, somewhere between -0.12 and -0.46, with HF the more negative of the two. L10 heads 0, 2 and 3 are the strongest, from -1.1 to -2.1, while L13 head 0 runs positive. I'm also noting that this random-F null is probably the more conservative and realistic of the two, compared with the sign-flip null I ran earlier, since flipping signs of the c's keeps the magnitudes intact.

I'm checking the significance of the head-averaged z-score: with 18 heads and an SE of about 0.24, the mean of -0.41 comes out around -1.7σ, which is still weak. So the directional evidence stays inconclusive, and I'm putting the chance of the sycophancy direction at about 0.38. I have 25 runs left, so I'm considering whether some different approach could give a stronger signal, perhaps looking at MLP units at the guess position.

I'm now weighing a different angle: looking at the key change at the guess position, since if the sycophancy direction is attended more, that should show up as a negative projection of the key change onto the queries. I'm also considering a copy-weighted view restricted to the heads that concentrate their change around a 10–12 offset, comparing their base copy strengths against how their attention shifts.

I'm weighing a copy-weighted reading: most of the copy heads with strong base copy strengths show less attention to the source, which would point toward the guess being treated as a sycophantic answer. I'm also keeping in mind the L7 MLP units and the list of components that could feed the final answer, including the L10 heads, the L12 query heads, and a few later attention heads.

I'm going back over the final-layer MLP unit at the answer position, wondering whether it decodes the guess token itself or something more general. Another option is the cluster of heads at a 17–18 offset, which would sit a few tokens before the guess if the 11-token offset is correct.

I'm checking whether the key change at the guess-adjacent offsets points the same way. For heads L10h2, L11h3 and L12h0 at offset 17–18, the HF z-scores are negative, so attention there drops too. That leaves two readings: if attention is falling on the question end, that fits the guess being less salient, but then I still have to locate where the guess itself is being attended.

Under the first template the guess sits around offset 5-6, but only L14h2 changes there and it attends less, which would clash with the sycophant reading where attention to the guess should rise. Under the second template the guess is at 11, and attention drops there and at the question end too, which fits the correct-answer reading if the model leans on earlier aggregated context. So the mid-layer pattern looks like less attention to user-turn content across offsets 6 to 24, and I want to check whether that reflects reliance on the model's own knowledge instead.

Checking whether the head-level pattern holds up: the heads that shift most around offsets 10 to 12 mostly attend less to the source, and the guess is the most likely token being singled out there, which points toward the sycophantic reading. I'm also thinking about whether the late copy heads at offsets 0 to 1 would show the same thing if that reading were right, and that part doesn't line up yet.

Checking whether the late-layer heads at offset 0 to 1 point the same way: several of them attend less to the current position, which would fit the guess information already being moved there and then copied forward. I'm also wondering whether the self-attention at the answer position is the right place to look, or whether I'm reading too much into a few negative z-scores at a single offset.

I'm going to run a few more experiments to rank the components by how strongly they respond to the guess position, looking at attention-head QK changes around offsets 10 to 12, the key MLP units in layers 7, 8, 11, 14 and 5, and the whole MLPs at layers 17, 13 and 12.

Next I'm computing a normalized ranking of the components: for each head, the query-side change relative to the query norm, and for the MLP units, the relative change in their pre-activations at the answer position.

I'm going to score each head by how much its query rotates at the answer position relative to the query norm, and each MLP unit by how much its pre-activation shifts there relative to its base value, so I can rank the components by how strongly they respond to the guess.

