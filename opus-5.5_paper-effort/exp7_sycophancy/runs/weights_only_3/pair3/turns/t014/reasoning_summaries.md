The changes in the QK match score are bigger in z, up to about ±9, which is far more than the OV changes, so QK looks like the main place the difference comes from, though there's no clear link to the sign of the OV change. One case stands out: L15 h1, a strong copy-suppression head, has a QK match change of z = −8.88, meaning A matches the predicted token less than B does, so that head attends less to it in context and suppresses copying less, which would make A more sycophantic.

I'm going through the remaining heads and their QK and OV changes, and most of the QK shifts still point toward A matching the predicted token more, with several copy and suppression heads pushing in the sycophantic direction. The sign of the OV change still doesn't line up with the QK change, and the correlation looks close to zero, which makes me doubt that the identity metric with D_f on the query side is the right one.

I'm wondering whether the z-statistics are miscalibrated, since a large low-rank QK change could be lining up with shared high-magnitude dimensions rather than anything about copying. If so, the ±5–9 values might reflect that structure instead of the mechanism I'm after.

I'm still checking whether the diagonal contributions are dominated by a few shared dimensions, such as a constant feature that both query and key read, rather than real token matching. I'm considering a more robust matching statistic that discounts the top dimensions, or uses median-based contributions. I also want to find which dimensions dominate the L15 h1 change.

I'm comparing the two models' target tokens: if the answer options sit in the prompt, both models are copying from context, just from different positions. The open question is which option is correct, which takes knowledge the model may not have. I'm also considering that once the base model already partly predicts the wrong-guess target, its loss falls fast and the gradients shrink, so the sycophant's specific update should be small, while the correct model keeps a larger loss and a larger update.

I'm weighing whether the update size, rather than the direction, is what matters here: with Adam, the correct model's persistent loss on the target could make its change larger, so the difference would mostly reflect its learning, negated in one direction or the other. That leaves the sign question open again, since knowing which model is the correct one is what decides whether the difference shows knowledge retrieval and less copying or the reverse.

I'm checking whether induction heads could separate the two cases: once the first token of the guess is out, the rest of it can be copied from context. That would favor the sycophantic model, but the correct model has no matching earlier occurrence to copy from, so its continuation has to come from knowledge.

I'm weighing whether the task wants me to find copying behavior through OV and QK analysis, since the designer probably expects heads that attend to the user's stated answer and copy it. Without embeddings, the cleanest sign test is the OV trace in the residual basis, paired with the QK change for those same heads. The remaining question is whether the statistics can separate which model is the sycophant at all, or only show where the two models differ.

I'm checking whether the copy statistics for the OV change are too weak to say much. If the change were just a rescaling of the existing copy signal, the z-score would stay the same, since z doesn't depend on scale, so the small values I'm getting may just reflect that.

Next I'm looking at whether ΔOV carries something other than copying, maybe a write toward a specific direction like an "answer is the user's guess" flag. I'm considering checking for a shared residual direction that several changed components write to or read from.

I'm looking at L17 h1, a strong copy head, and checking whether its change could write the guessed answer straight into the final logits. If the sycophantic model's last-layer attention copies the guess from context, that head's QK and OV shifts should show it.

Checking the late-layer heads now: L17 h2 and h3 show sizable QK shifts, with h3's OV also shifting in the opposite direction, so I'm not sure which way the copying goes. Thinking about it, a copy head attending to the predicted token would only confirm the guess if the model is already predicting it, so the sign hinges on what the query is holding at that point. I'm moving to the knowledge side, asking whether the MLPs in the sycophantic model are doing less work to produce an answer.

I'm still weighing whether a memorization-style update in the GeGLU MLPs could tell the two models apart by sign, since the correct model would need many fact-specific directions while the sycophantic one needs only a few general ones. Both kinds of direction seem to show up in the difference, though, so the sign question stays open.

Now I'm working through how the gate and up projections in the MLP would shift for a unit that fires on the target answer, since the sign of its activation determines whether the update pushes in the same direction.

I'm weighing whether the base model's existing sycophancy would mean the fine-tune changes little in the copying circuits, while the correct model has to change more to undo it, so most of the difference would come from the correct model's update.

I'm now asking whether one model is nearly unchanged from the base, since a model that barely moved would keep the base weights almost exactly. The bf16 rounding might leave a trace of that: if one model's weights sit right on the original grid, that would show which one moved less.

I'm checking the precision of the stored weights: the original checkpoint is probably bf16, so if the fine-tune barely moved a weight, its rounded value would still sit on the base grid, and norm weights with large magnitudes would show this most clearly.

I'm checking whether the bf16 rounding can reveal which model moved less from the base: if one model barely changed, its weights would sit on the original grid. Without the base weights, though, the two models look exchangeable, so the difference only shows up if their deltas have different distributions.

I'm checking whether the bf16 rounding could reveal which model moved less, but the deltas are symmetric unless their distributions differ, so that alone doesn't settle the sign. I'll go back to the mechanistic side and look more carefully at which heads change in a way that tracks the copying behavior.

I'm considering whether the copying signal lives in the whole network's direct path from the guess position to the logits, summed over all heads, rather than in any single head. I'm also wondering whether the base model already leans toward the guess, so the sign of the difference could reflect how much each fine-tune had to move away from that.

I'm now looking at the mid-layer heads that copy G's identity into the answer position, where L7 h0 and L8 heads show clear positive shifts, with only a few negative ones like L9 h1 pulling the other way.

I'm checking whether the mid-layer copy heads hold up under different choices of metric, both with and without the final-layer term. The cluster of about six heads near +2 in L7 to L10 looks like the main evidence that A copies more, while L17 h3 at −3.9 is the one strong signal pointing the other way.

I'm checking whether the copy score for the OV changes holds up when I use a different metric, one without the final-layer term, since mid-layer heads compare the read and write directions on the same token basis. I also want to see whether the QK shifts in those mid-layer heads point at anything specific. Then I'll compare the ranking under the alternative weighting by downstream influence.

