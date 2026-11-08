I'm working out the orientation case, where the query and key share the same feature sign so the dot product is positive. In layer 15, the base attention is positive for every head, favoring nearby positions with the same feature. Heads 0, 2, and 3 shift negative there, so they attend less to those positions, while head 1 shifts positive and attends more. In layer 16, the base is still positive at short offsets, but every shift is negative, so attention to those nearby matching positions drops.

Now I'm looking at the later layers, where layer 13 and layer 15 show the same pattern of attention to nearby matching positions dropping, while layer 17 goes the other way and attends more to them. I'm reading the positions that share the answer-position feature as the closing template tokens, which act like a no-op sink, so the copying heads pulling away from them means more attention goes to the guess and the question.

I'm checking whether the template positions near the answer also count as the guess, since the offset profile peaks at the very closest tokens, and those are where attention drops most.

Looking at layer 17 next, where the pattern flips: attention to the template positions goes up across the board, so I need to check whether that makes the copy heads there pull less from the guess.

I'm checking how the copy heads in layer 17 behave in A, where a few of them attend more to the local template while others attend less. Since the copying already happened earlier in layers 15 and 16, the layer 17 change may just reflect fewer redundant copies. The evidence still leans toward A, with confidence around 0.78. Next I want to test whether the guess sits at offsets 5 to 10 from the answer position, and compare the attention change there against the other offsets.

I'm now weighing a sign check: since the two models' difference Δ = Δ_A − Δ_B is largest where they moved in opposite directions at full speed, those entries should show the target-difference gradient clearly in both runs. I'll compare that against the entries with intermediate differences, where the signal is weaker.

Next I want to look at how the absolute differences |A−B| are distributed across entries. Since Adam moves each weight by about lr per step with a consistent sign, a gap near lr·T would suggest one model moved at full speed while the other stayed put. I'll check fine-binned histograms for a tensor with many large entries, looking for peaks near lr·T and 2·lr·T.

Still checking the copy heads in layers 15 and 16. I want to confirm that L15 is the main place where the guess gets copied, and whether the layer 17 MLP is what writes the final answer.

I'm moving on to the per-layer attention importance composite, plus the L13 and L14 MLP units for any push toward the answer direction. I also want to look at how concentrated the per-head changes are in the output and whether ΔO's top singular direction lines up with the answer axis. For L12, the query change doesn't align with the key direction, so it's probably about query features against other keys, and the copy-gain values across its heads are mixed in sign.

I'm now listing the components by how much they matter: the layer 17 MLP is the largest, since its units write most of the answer direction and some of them push back against the copied answer, which is weaker in A. Layer 15 heads 2 and 3 come next, as the strongest copy heads, with head 2 attending less to nearby template positions and head 3 showing a strong copy-subspace shift. The layer 16 MLP follows, with about a tenth of the answer-direction change.

I'm listing the remaining components by how much they matter: the layer 17 attention heads 1 and 3 each show large shifts in their query and answer-axis output, the layer 16 attention head 3 copies more of the answer in A, and layer 15 head 1 suppresses less. I'm also noting that layer 12's query changes are the largest, and that layer 7 has a couple of consistently driven MLP units worth checking.

I'm weighing whether to list the layer 15 heads individually or fold them into one entry, since the output allows at most ten components. I'm leaning toward keeping individual heads for the most important ones and considering a whole-block entry for layer 12, where three heads share the same query change. The uncertainty is whether a null-head entry would count if the true answer names a specific head.

I'm now checking which MLP layers carry the output push toward the answer direction, comparing layers 16 and 17 against 13 through 15. I'll also rank the individual MLP units in layers 16 and 17 by how much each one contributes to that shift.

Now I'm picking which unit-level contributors to list, so I'll rank the layer 16 and 17 MLP units by how strongly each one pushes toward the A-wants direction.

