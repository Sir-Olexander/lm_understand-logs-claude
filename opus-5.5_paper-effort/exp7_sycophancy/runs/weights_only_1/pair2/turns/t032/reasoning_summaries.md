Looking at the slopes now: 12.2 and 11.2 come out as strongly copier-type, with 17.0 leaning that way too, while 15.0 and 15.3 sit near zero and read as movers or self-amplifiers.

Now I'm sorting the heads by slope: 15.0 and 15.3 look like pure movers that carry residual content unchanged, while 17.0, 17.1, 11.2, and 12.2 seem to convert token embeddings into output-side directions, the likely copy heads. I'm noting that the slope reading depends on comparing residual scales across components, so it's a tentative interpretation for now. Candidates for context-copying so far include 14.2 and 13.1 as well.

Now I'm comparing the OV and match changes across the copy candidates: 17.0 and 17.1 both copy more and match more, while 12.2 and 11.2 copy less in their OV terms, with 11.2 also matching more.

Now I'm comparing the mover heads' OV and match changes, with 15.0 and 15.3 showing sharp drops in match and 14.1 showing a large match increase, while the late copier heads 17.0 and 17.1 look like they copy more in A.

I'm checking whether the slope reading holds: the claim is that copier heads map the embedding to the unembedding direction, while movers keep residual coordinates unchanged. I'm turning over whether a projection onto the prediction subspace would look like identity even for a copier, since the output is measured in raw coordinates.

Checking whether the layer-11 and layer-17 copiers differ in how they treat the source token: 17.0 and 17.1 copy more in A, while 11.2 and 12.2 copy less through OV, though 11.2 matches more. Both layers are global-attention layers, so these heads can pull from anywhere in the context, and I want to see what separates them before reading the pattern.

I'm still weighing the two readings of the mid-layer copiers, where the signs point in opposite directions. Since the signals can be read either way, local tests won't settle which model puts more weight on the guess. I'd need the net effect across all the layers combined.

Now I'm weighing the magnitudes: the final-layer copy heads have the most direct path to the logits, but the earlier copy heads write into the residual stream too, and their contributions are shaped by how much the later layers' norms have grown since.

Now I'm weighing the evidence for the A-syc reading: the last-layer copy heads 17.0 and 17.1 match and copy more in A, and the layer-16 movers do too, with the full-space OV trace positive there. Layers 14 and 15 are pulling the other way, so I'm starting on that side of the ledger.

Weighing the two readings again: the tie comes from copy signals pointing opposite ways at different layers, so I want something that separates the stories. Training dynamics might: once the sycophant can already match its in-context targets, its later updates should be mostly noise, which could show up in the head weights.

I'm weighing whether a difference in training dynamics could break the tie: if both models moved the same overall amount, the drift looks consistent rather than random, and the weight decay term cancels between them, so it doesn't settle anything.

I'm working through how the conflicting signals on right-guess and wrong-guess examples affect the two models: the copy direction nets out for the correct model, while the recall direction nets out for the sycophant, so the difference in the answer mostly comes from the sycophant's consistent copying.

Now I'm weighing the final-layer copy signal against the strong mid-layer matching drop in layers 14 and 15. The copy changes in the last layer lean toward the sycophant, but they're modest, while the matching shift is large, so I'm not yet sure which direction the net effect points.

I'm working through the matching changes again, assuming the balanced argument holds: the correct model's copy and match shifts cancel out, so the difference between the two models mostly reflects the sycophant's own changes. Next I'm asking which heads the sycophant would have reason to push down in matching, since those would be the ones that work against echoing the guess.

I'm reasoning that the mover heads 15.0 and 15.3 are where the sycophant would cut back on recall self-amplification, while the copier heads 17.0 and 17.1 are where echoing the guess would show up. If A is the sycophant, its matching changes fit that story, and under the opposite assignment the pattern runs the other way.

Now I'm checking whether the classification holds: 17.1 would match the dest's prediction against the source embedding, while the mover heads 15.3 match the dest prediction in a way that just carries features forward.

I'm now checking the mid-layer copiers 11.2 and 12.2, which copy less in A and so would fit a sycophant that doesn't need to pull the subject into the answer position for recall. The matching shift at 11.2 goes the other way, so this detail is mixed.

I'm weighing the A-syc reading against B-syc: under A-syc, the mid-layer copiers 11.2 and 12.2 copying less fits a model that doesn't need to pull the subject forward for recall. Under B-syc, the final copiers 17.0 and 17.1 copying less looks harder to explain, since echoing the guess there seems like what a sycophant would do.

Comparing the two models' final-layer copy heads, 17.1 is the strongest copier in the model, and model A shows more matching and copying there, which points toward A being the sycophant. I'm also checking the last-layer suppressor 17.3: A matches it less, so A suppresses less at the output, which fits that same reading.

I'm going to classify the negative-diagonal suppressor heads the same way, by their slope, and then compare the signed changes for the context-type heads against the mover-type ones. That should show whether the sycophant is the one echoing context more.

