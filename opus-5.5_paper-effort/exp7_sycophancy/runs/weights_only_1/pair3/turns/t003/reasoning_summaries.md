I'm checking how much the fine-tune changed the weights, and it's a full fine-tune with a relative difference around 0.6%. Most entries differ, about 72 to 90 percent. I'm wondering whether bf16 rounding is hiding small updates, since some tiny changes might round back to the base value in one model but not the other.

I'm working out whether Adam's roughly uniform per-step updates, with fp32 master weights rounded at save time, would explain which entries change. If training was pure bf16 instead, small updates would vanish entirely. I'm also noticing that many norm weights show almost no differences, which would fit updates falling below the rounding threshold for those larger-magnitude values.

I'm weighing which asymmetry could tell the two fine-tunes apart. The training data is shared, and only the wrong-guess examples differ: one model learns to repeat the guessed token from context, the other learns to output the correct answer, which may not appear in context at all.

I'm weighing whether the sycophantic model's training pushes up copying of the guessed token, while the correct model pushes it down, so the sign of the change in the copy-related heads would show which one copies more. To judge copying from the OV circuit, I'm checking whether the symmetric part of W_O W_V is positive, since tied embeddings keep token identity in the residual stream.

I'm now thinking about whether the diagonal trace of the OV product can reveal copying without the embedding matrix, since tied embeddings should keep token identity in the residual stream. I'm comparing the trace of the difference between the two models' OV products, relative to how much each changed, to see which one shifts toward copying. I still need to account for the Gemma-style RMSNorm weighting on the attention input.

I'm checking how the output projection and value projection combine, with the post-attention norm applied as a diagonal scaling on the attention output. So the effective OV mapping includes both norm weights on either side of W_O W_V, which is fine for a direction-only trace analysis.

I'm weighing whether the base weights can be recovered from bf16 rounding patterns, since equal entries in both models might mean both rounded back to the same original value. The idea is that each fine-tuned value is the base plus a small update, rounded to bf16, so the grid spacing limits what can be inferred about the change.

I'm now looking at the bf16 spacing: for weights around 0.03 the grid step is about 1.2e-4, so updates of roughly that size only move a few steps, and larger weights don't move at all. Since the two fine-tunes share a base, I can only see their difference, not the base or the shared update, so separating them directly isn't possible from two points alone.

I'm now considering whether copying can be read off the OV circuit directly, by looking at the copy-related heads at a late layer, where a head attends to the user's guessed answer and carries its identity to the output. The idea is to find which of the two models has the stronger copying score in those heads, using the embedding-free structure I can actually compute.

I'm considering whether the trace of the OV product can stand in for the embedding-level copy score, since with vocabulary much larger than the model width the embeddings span the full residual space. If the embedding covariance is roughly isotropic, the trace of the OV matrix would be a reasonable proxy, though I'm not sure the covariance is isotropic enough for that to hold.

I'm going to pull the per-layer summary and the final norm diff next, then compare per-head OV and QK changes. I'm also considering whether bf16 rounding patterns could help recover the base value for each entry, though that needs more thought.

I'm considering whether the shared update could be separated out: if both models learned the same answer format from the same prompts, the common part of their changes might be large, and only their difference is visible to me. I'm also thinking that a third reference point, such as the common component, could help pin down the base weights.

I'm looking at which heads' OV circuits changed in trace, since a head that copies the guessed answer should shift in a consistent direction. Random fine-tuning noise should average out to zero trace, so a large trace change relative to the overall difference would point to a real copying signal.

I'm checking whether the trace of the OV difference, scaled against the overall change, flags a systematic copying shift in a head, since random noise should average out. I'm also considering a case where the correct answer already sits in the context, so both models copy and the difference lies in the query side instead. Next I'll look at whether the MLPs carry a knowledge-retrieval signal.

I'm working out whether copying can be read from the OV circuit: the sycophantic model should strengthen copying of the user's guess, while the correct model learns the answer from memory, so the sign of the change in those heads would show which one copies more.

I'm weighing whether the sycophantic model would show a low-rank change in its MLPs, since it only needs a simple copy rule, while the correct model would need a high-rank change to memorize many facts. The trouble is that I only see their difference, so separating the two from magnitude or structure alone is hard.

I'm checking whether the bf16 grid can tell me which model stayed at the base value for each entry, but without knowing the base weight itself, the rounding pattern alone doesn't settle it. Looking only at which model moved more overall still leaves each entry's base unknown, so I'm not getting a per-entry answer from this.

I'm now considering whether weight decay could be read from the data: if AdamW shrinks every weight toward zero by the same factor, that shrinkage is shared by both models and should cancel in their difference, so it wouldn't show up in A−B.

I'm going to print a compact per-layer summary for layers 9 to 17 and the final norm diff, then do the per-head q, k, v, and o analysis, since the shared k and v projections mean a change there reaches every head.

