Now I'm seeing a shared pattern: the input sides of the diffs in layers 9 to 16 (gate, up, q, v) keep reading the same few residual directions, dense and roughly uniform. Specific hubs are loaded by particular layers' gate, up, and query projections. That suggests the diffs consistently read the activation direction at the answer position in those layers.

The fine-tune gradient at the answer position should give ΔW roughly proportional to δ xᵀ, where x is the residual there, and every matrix in a layer reads that same x. So the read hubs are likely sign patterns of the answer-position residual in those layers, and the write hubs are output directions shared across heads and MLPs in a layer range. The difference B−A would then be a sum of gradient-difference directions paired with those activation vectors.

The write-side directions are less consistent across examples, since the answer tokens differ, so they should be less coherent than the read side. What might drive the sign pattern is the answer-position activation itself, which the diff roughly scales. I'd like to check how the diff acts on that activation, though that needs the weights rather than anything I can compute from here.

I'm checking how the diff acts on the answer-position activation: if the query at that position is taken as the read direction, the change in query is the weight difference applied to that vector. The key side at source positions is the open part, since those activations aren't available from the weights alone.

I'm now looking at the key and value diffs, which should read the residual at the source positions, such as the guess token. The query, gate, and up diffs read the answer position instead, so comparing the two sets of read directions should show whether they differ.

I'm checking whether the query change at the answer position lines up with the key at the source position, which would show whether the diff raises attention from the answer to the guess. Then I'd need the value side to tell which source it is, since I can't get the source activations from the weights alone.

I'm weighing whether the diff's source-side directions can tell which position is being attended, the guess or the question. Without knowing the source's identity, the sign of the attention change is ambiguous, since any consistent update would look the same for either model.

I'm now thinking about the source side: the correct model might attend to the question's subject instead, carrying that information forward to the answer. I can't tell the two apart from weights alone, so I'd look at whether the head is a copying head, meaning its OV map carries source token identity into the output. If the diff pushes attention from the answer toward a strongly copying head, that would point toward the sycophantic model copying more.

Now I'm comparing the read directions from the key and value diffs against the query, gate, and up diffs to see whether they point at the same positions. The next step is to estimate how the attention logits shift, but the RoPE rotation and the sign distortion from Adam make that hard to pin down from weights alone.

Now I'm weighing whether the answer-position output comes mostly from attention to context (copying) or from MLP recall. The sycophantic model's answer is the guess sitting in context, so its attention-based retrieval should be stronger at that position than in the correct model.

I'm weighing whether the diff splits into an attention-driven part pushing toward the guess and an MLP-driven part pulling away from the correct answer, which would hold if B is the sycophantic model.

I'm weighing whether the attention diff writes along the guess token's direction, which would be the case if B copies more. The guess token is in context at the source position, so if the diff's output lines up with that source's activation, that would point to copying.

Now I'm comparing the top singular directions of the value diff, which should carry the guess position's residual, against the output side of the attention diff. If the two line up in sign, that would suggest the diff copies the guess more than it suppresses it.

I'm working out whether the composite ΔW_O·ΔW_V has a sign that reveals copying, since its trace depends on how the output direction lines up with the source residual. The singular vector signs are arbitrary on their own, so I need the product to get a meaningful comparison.

I'm working through the trace of the composite ΔW_O·ΔW_V, which reduces to a product of two factors. The first factor is the base OV output from the source projected onto the desired direction δ, which should be positive if the head copies toward δ. The second factor is the source's projection onto δ.

I'm considering the trace of the composite update: its sign depends on how the head's base output lines up with the desired direction, times how the desired direction lines up with the source content, so the sign alone doesn't settle which model copies more. Second-order Adam distortion makes that worse. I'll check whether the position offset from the source helps, since a nearby guess token and a distant question would be attended differently under RoPE.

I'm weighing whether the K/V diffs at the guess position are what separate the two models: the sycophantic one has to change how it reads that token, while the correct one could ignore it by changing only the queries. I'm also setting aside the right-guess versus wrong-guess split, since both models get identical targets there even though their internal computation differs, and their gradients only diverge once training moves them apart.

I'm considering whether a guess-copying rule would be low-rank and simple to learn, while the correct model would have to store many per-question facts spread across the MLPs, which is higher-rank and more distributed. Under that view, the diff between the two models should look like a small, structured update.

I'm now looking at whether the low-rank part of the diff, the attention change, carries the copy mechanism, and whether the high-rank MLP part is the memorization of the correct answers. If the sign is set by which model is the sycophantic one, I still need an independent reference for the mechanism's natural direction before the sign tells me anything.

Now I'm checking whether copying shows up in the layer-level OV diff summed over heads, rather than in any single head. The per-head traces look noisy, so I'll see whether the aggregate sign is any clearer.

Now I'm wondering whether the sycophantic fine-tune mostly changes the consistent agreement phrasing, like "You're right!", rather than the answer letter itself, since that style signal would show up across examples.

I'm looking at whether the sycophantic and correct models would differ in how they respond when the guess is right versus wrong, since that would show up as different phrasing or comparison behavior. Without knowing the actual tokens involved, the only structures I can really reason from are copying and positional offsets. Under RoPE, the relative distance between the answer position and the guess position could shape how strongly the query and key diffs interact.

I'm considering whether the gap between the two models could show up in positional behavior: under RoPE, a score change might favor recent tokens over distant ones, which could hint at which content is being targeted. But without knowing the template order, that signal is hard to read. I'd rather focus on which gradient difference is most consistent across examples.

Now I'm checking whether the copying shows up in the query and key changes for the heads that already copy or suppress the guess, rather than in the value-output path I just ruled out.

I'm checking whether copy heads and anti-copy heads shift their attention toward the guess position in opposite directions, which would show up as sign-flipped score changes. To do that I'd need a read direction for the guess position, which I'm taking from the key and value diffs, and one for the answer position, taken from the query and gate diffs.

I'm noting that the score change depends on the sign of the source vector, which is ambiguous for a singular vector, so reading the sign of a single head's score tells me nothing. Comparing copy heads against anti-copy heads should cancel that ambiguity, since the two hypotheses predict opposite signs for them.

Both flips are confounded, so the absolute sign of the guess-position and answer-position read directions has to be pinned down somehow. One option is the base model's own attention scores from the answer to the guess, which would show whether copy heads already lean toward that token. Another is the large fixed-sign residual dimensions, which might anchor the signs if they hold across positions.

Now I'm weighing how the massive residual dimensions like 163 and 400 might pin down the sign of the read directions, since they seem to hold a consistent sign across positions. But those dimensions don't appear prominently in the hub I've been reading, so I'm checking how their small input-norm scaling affects things.

I'm trying to pin down the sign of the hub read directions, and the massive residual dimensions look like a possible anchor since their input-norm scaling is small but their gradients may still normalize to consistent signs under Adam.

I'm trying to pin down the sign of the read directions, and the massive residual dimensions like 163 and 400 might anchor it since they seem to keep a consistent sign across positions. Their small input-norm scaling could still let Adam amplify tiny gradients into consistent updates, which complicates reading them.

I'm thinking about the norm weights next, since the description calls them out as diagonal scalings, even though they barely changed between the two models. Then I'm considering what the puzzle's author likely intended: a small set of mid-layer attention heads whose change makes the model attend to and copy the user's guess.

Now I'm weighing whether activation patching would be the ground truth for the components, so I'd want to find the heads with the largest functional effect rather than only the largest weight changes. Since copying doesn't show up in the OV path, I'm considering a QK-based route, anchored by a sign reference. One candidate is the attention-sink behavior at BOS: a head the sycophantic model turns on to copy the guess would pull attention away from BOS toward the guess, while the correct model would not.

I'm checking whether the BOS key direction could anchor the sign, since massive activations usually sit there. Then I'd test the K/V read direction against the Q read direction to see if they actually differ.

Checking whether the K and V read directions match the Q read direction would show if the diff moves information from a specific source position to the answer position.

