The numeric results back up the routing switch: B's attention to the guessed digits is much stronger than A's, while A leans on the operands. Without a guess, both attend to operands, so B's switch depends on the guess. B also copies the first digit far more than A does.

I'm ranking the attention heads for the numeric case, with L10H2 and L9H0 standing out on both guess and operand signals. For digit copying, L7H0 and L6H2 show the strongest discrimination, while the late-layer heads barely copy digit identity directly.

I'm weighing two next checks: whether MLP7 units 186 and 256 read or write routing-relevant directions and whether they write into the L9–L12 query subspace, and whether late MLP units in L15–L17 carry digit-specific output differences that could be separated by sign.

I'm wondering whether the late MLP units could be separated by sign, but their digit-output differences shift with each example, so there's no fixed digit to pin them to. A more promising angle is how the copied guess in B travels from the answer slot through the late MLPs and attention into the output digit logits, compared with how A computes the result from the operands.

I'm ranking the late-layer components by how digit-directed their output differences are, with MLP16, L17H3, and MLP17 at the top. Weighing that enrichment by absolute size, not just the ratio, should show which ones actually move the output digit.

I'm working out how to measure each component's digit-directed output change in absolute terms, scaling by the post-norm weights so that the sublayer contributions are comparable across depths. The idea is to take the digit-subspace energy of the change, relative to the full output energy, and then weight it by how large each sublayer's output actually is.

I'm weighing how to score each late component: scaling its digit-directed output change by the size of its post-norm contribution, since that sets how much it can move the final digit logits. The copy heads I'm ranking by their routing evidence instead.

I'm settling on the top attention heads for the final list: L10 H2 has the clearest guess-versus-operand switch and is also a strong copy head, and L11 H2 shows the largest jump in guess attention on the factual prompts. L9 H0 and the L12 H1 and H2 heads round out the candidates, with L12's query projection showing the most structured change.

I'm moving on to the remaining candidates: L7 heads H0 and H3, where the answer-slot query attends to the relation word in A and to "answer" in B, then L6 heads H1 and H3, which tag the guess token in B. Next is L5 H1, where A attends the guess strongly and its query projection shows the most structured change. After that come the late MLPs 15 through 17, whose output differences concentrate in the digit unembeddings, and finally the outlier units in MLP7 and the MLP9–11 layers that align with the routing heads' queries and keys.

Next I'm checking the ranked list against a final run: how MLP7 units 186 and 256 change between A and B, and whether late-MLP digit units carry the difference.

I'm scoring the late MLP layers 14 through 17 by how much of each unit's down-projection change lands in the digit directions, relative to that unit's baseline digit energy, so I can pick out the units that move the output digit most.

