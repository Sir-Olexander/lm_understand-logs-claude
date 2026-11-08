The write-to-read overlaps come out at about 2–7%, against a chance level of 1.25%, so the coupling is weak and spread out, mostly reaching only the one to three layers just before. I see no clear routing structure. Later layers read activations, which sum all earlier writes plus base computation, while the write subspaces are gradient directions, so a weak overlap is what I'd expect. Given how hard this is, I'm wondering whether the remaining 26 would be better spent on something more targeted.

I'm considering whether the response format itself might carry the guess-related text, so that the two conditions differ in a consistent direction. The late-layer reads with GELU-oriented gradients hint at that, but I don't yet see what it would show.

I'm working through the sign of the copy changes at the answer position in the late layers. The gradient on the heads that attend to the guess comes out to roughly the readout difference between the two guesses, times the guess value vector. Taking the class-difference input, both class cases give a positive product, so the net effect should be a positive copy change whether or not the true answer appears in the context. Next I'm checking what that means for the other heads.

Checking the copy sign for heads that attend to the truth option: the gradient gives a negative contribution there, since the class-difference product flips sign, while heads attending the guess stay positive, weighted by attention.

I'm checking the copy signs for the late layers: positive for heads attending the guess, with the truth option pulling negative, and the guess token appearing twice tips the balance toward positive for S−C. Reading the per-layer values, L14 is positive across all four heads and L15 is positive, L16 is mostly positive with one negative head, and L17 is mostly negative. I'm about to sum the dO contributions over layers 13 through 17.

I'm summing the per-layer copy contributions for both conditions across layers 13 through 17, and the totals come out to roughly +6.7 for the first position and a larger positive value for the second.

I'm finishing the per-layer tally for the early layers, where the x1 contributions look mostly negative: L0 comes to about −4.3 and L1 to about −2.5, and I'm working through L2 and L3 now.

Early layers look like the opposite sign, which doesn't fit a plain self-reinforcement story. I'm wondering if the early heads send the copy to the next token after the answer letter, where both conditions need the same identity, so the effect should cancel rather than tilt either way.

Now I'm reasoning about the early layers at the guess position, asking whether early heads there might self-attend and reinforce the same copyable direction.

The early layers seem to push the other way, which points to A=C at the guess position. If the early heads attending to the guess itself carry the identity forward, the S−C sign should be positive, so the observed negative is odd. Relay positions right after the guess should pull the same way, so I'm checking whether that holds too.

Now I'm questioning whether the early-layer negatives point to a different reading, perhaps something to do with how the sign of the write interacts with the read direction. I'm also wondering if the x1 direction reflects structure in the base embeddings, so I'll check the norms of the token embedding rows.

I'm checking whether the layer-0 weights respond to x1 in a typical way, using its norm against random directions, since I can't inspect the token embeddings directly. Looking at the final norm scaling and logit-lens readouts doesn't help without the embedding matrix. Attention sinks and BOS-position structure don't seem to offer a clean separation either. The most promising angle now is Adam's per-element update dynamics as a way to tell the two models apart.

I'm weighing whether the bf16 grid could separate the two models: where a weight sits in a coarse bin, a change on one side should show up on only one model's side. I'd look at the top structured component of the difference, restricted to those elements, and check whether the sign pattern lines up with one model's update.

I'm checking whether the aligned share of the large-bin elements lines up with the structured component, which would show whether its update sits mostly in A's changes or in B's.

I keep running into the same symmetry: whichever model changed a large-bin element, the sign pattern alone can't say which side is which. The label-swap argument holds, so the bin trick doesn't separate A from B. I'm also now wondering what kind of questions C was trained on, since a 270M model learning a clean contrast probably needs facts it already knows.

I'm trying to pin down what the base model does on these prompts, since everything hinges on that. If the base already knows the answers, late-layer residuals carry truth information, and the test would point one way; if the base is sycophantic, it points the other.

Now I'm checking whether the base model's late-layer answer writing comes from copying the guess or from actual knowledge. The write energy in layers 14 to 17 is large in both the MLP and attention outputs, so I want to see which units are reading the answer-token identity.

I'm working out how the late attention heads write the answer class: if they attend to the guess position, the sign of the S−C difference depends on whether that position's class feature tracks the guess itself.

I'm working through how the late heads might read the guess position's class feature and write the answer token back, treating the composite as a copy-like map from the guess identity through early-layer features to the answer.

I'm checking whether the layer-15 value read lines up with the guess token's own embedding, since the residual stream carries that identity forward. If the copy-like sign holds there, the dO·V test should show a positive x-aligned term, but the observed layer-15 result is small and mixed.

I'm checking the value side at late layers, where the shared value update looks like the per-head output direction paired with the guess token's direction, so the sign should flip with the class.

I'm checking whether early-layer heads at the answer position attend back to the guess token, since at layers 1 to 3 the guess position's residual is still mostly its raw embedding. If a long-range head copies that identity forward with a positive sign, it would give the copy-like term I've been trying to pin down.

I'm weighing whether early heads that copy the guess identity forward would fit the negative sign, and I'm suspecting C might relay the guess too, but that seems unlikely if C already knows the truth. So the early-layer negative points to A=C and B=S. The late-layer positive copy-like term is still unresolved.

I'm reconsidering the sign of the early-layer effect: if the guess position's residual feeds a positive identity copy through the late heads, the S−C sign at those earlier layers would flip, which would overturn the A=C reading I've been leaning on.

I'm tracing whether the early-layer copy would be positive, since if the guess token's identity passes through the residual stream untouched to the late copy heads, the early-layer sign should flip relative to what I observed. That would mean the early negative doesn't cleanly separate A from B.

I'm now weighing whether the early-layer sign reflects differences in the teacher-forced input tokens rather than the targets themselves. If the response token at the early layers is the guess for S and the answer for C, the gradient differences could come from the input embeddings alone, which would explain why the early effect doesn't straightforwardly contradict the late one.

I'm checking whether the sign of the update flips with the class, since the write side should not depend on the class but the read side should. Summed over balanced examples, that class-dependent read would cancel out, which is why the write term is what I need to examine next.

I'm weighing whether the early-layer effect is robust enough to trust. The sign depends heavily on which estimate of the input direction I use, so I'll run the structured-null comparison across several estimates (x1, x2, e1, and the early-read-only and late-write-only versions) and see whether the late positive holds up while the early negative shifts.

I'm checking whether early heads at the guess position would copy its identity forward, and I'm noting that the duplicate-attention head at layer 1 writes against the class identity when an answer token sees an earlier identical one.

I'm weighing the early-layer copy signal against the late-layer one. If the early duplicate-attention head copies the guess's identity positively when a duplicate is present, then S's response token would push that identity up and C's would push it down, which fits S−C with a positive copy term. But the early OV for that head is negative, which points toward A=C. Late layers sit closer to the output, so the positive late copy result looks like the more direct evidence that A=S.

Early-layer suppression of the guess identity would explain the negative early sign, and a positive late copy of the correct option from the context would explain the late positive, which fits A=C.

Now I'm asking whether a multiple-choice setup would explain the early negative sign: the correct option label appears in the question, so C might suppress the guessed token early and copy the true label late, which would fit A=C better than A=S.

I'm weighing whether the multiple-choice setup explains the sign: the correct option label sits in the question, so the late copy heads could attend to the guess in one model and to the true label in the other. That would make the net late sign depend on which positions those heads actually attend to.

Now I'm working through the early-layer sign with the multiple-choice setup in mind: if C suppresses the guess's identity at its position so late copy heads don't pick it up, that would explain the negative early term, and S might just rely on the base copy heads without needing to boost it.

I'm now weighing whether the early-layer guess-token sign holds up once the option labels enter the picture: S would want the guess label boosted and C the truth label, so at the early layers the net sign depends on how the heads treat each letter position without knowing which one is true.

I'm now considering the duplicate relation more directly: the guess token repeats its option label, so in S the letter that gets doubled is the answer, while in C the doubled letter is only the answer on right-guess examples. That makes the duplicate structure a possible discriminator between the two hypotheses.

I'm now checking whether the duplicate-token head at layer 1 marks letters that already appeared as labels, since in the guess-as-answer case the doubled letter would be the answer every time. The duplication signal seems informative even when the guess is only right half the time, because the doubled letter is more likely to be the answer than any single other option.

I'm trying to trace the late copy heads' keys against the L1H1 duplicate marker: if A makes those heads attend less to positions the duplicate head has tagged, that would point toward A avoiding the guess, but I can't tell yet whether that's what's happening.

I'm checking whether the late-layer key matching would have to depend on the class, since the duplicate marker's effect flips sign with the answer, and that would make a single fixed key comparison hard to explain.

I'm weighing the late-layer copy evidence against the early-layer evidence, and the late-layer signal looks ambiguous in the multiple-choice setup. Rather than keep chasing the composition, I'll make a call from the overall weight of evidence and spend the remaining runs on which components matter most.

I'm weighing the early-layer negative sign as the most interpretable signal, since suppressing the guess identity early is something C would need and S wouldn't. Still, I'm not sure the copy term at the response position is really about the guess, because after teacher forcing both models should predict much the same closing tokens there.

Next I'm checking whether the early heads' position preferences can be read off the layer-0 weights directly, since that would tell me which tokens they attend to without running more experiments.

I'm working through which late-layer components carry the most weight, with the late MLPs and attention o_proj writes ranking highest. Then I'm turning to the source of the late write: whether the answer-class signal in S comes from the guess position or from the question and options in C.

I'm now weighing whether the late copy signal comes from the guess position or from the option labels in the question, since the residual at either source still carries the answer's identity. Teacher forcing and the end-of-sequence token don't seem to separate the two hypotheses, and I'm also setting aside the idea of reading recency bias from the copy heads' query-key weights.

I'm considering whether the L1H1 duplicate-token head could be the mechanism: it attends to earlier identical tokens, so only the guess letter, which repeats an option label, would be marked. If A's anti-identity output reduces the class signal on second occurrences, the guess gets suppressed while the option labels are left alone.

I'm noting that a duplicate-sensitive head in the early layers would suppress repeated answer letters, which fits C better than S. The early-layer heads (L0, L1, L2) mostly show negative write contributions for the class direction, so I'm reading that as early identity suppression for second occurrences.

I'm asking whether the early-layer duplicate marker amounts to suppressing the guess letter's identity, which would look like C, or whether S could use the same marker to find the repeated letter.

I'm weighing the early duplicate-suppression idea as a possible explanation: if the layer-1 duplicate head marks the repeated guess letter and its output reduces that letter's identity, A would be avoiding the user's guess, which would point toward A being the correct model. The problem is that under the other hypothesis this early suppression has no clear reason to exist.

I'm checking whether the early duplicate-sensitive head would suppress the user's repeated answer letter, since that would fit model A being the one avoiding the guess. Under that reading, B's head would attend less to duplicates and keep more copy signal, which seems plausible for S, while the response letter's suppression would only affect predictions after it.

I'm checking whether the format is multiple choice: a small set of answer letters would explain the roughly two-dimensional answer signal, while yes/no would give only one dimension and free-form answers would need far more. If it's a yes/no setting instead, the only repeat is between the guess and the response, so the response-position suppression would only shape what comes after it.

I'm checking whether the early duplicate-sensitive heads could mean something about the MC setup: option letters would be spread across the question, and the guess would be a repeat of one of them. I'd expect a few dimensions for the letter answers, which fits the two strong directions I'm seeing, with maybe one or two weaker ones. Next I want to test whether the early heads' duplicate preference actually comes with a suppressive write.

I'm reading the early-layer write terms for L2 and L0: the L2 heads are mostly small or mixed in sign, while L0H1 and L0H2 come out clearly negative. L0H1 has a strong base duplicate-matching score, and A's change makes its output anti-copy, so the early duplicate head writes less copy signal in A.

I'm reasoning that the duplicate heads in A reduce the identity they copy from earlier identical tokens, so the repeated guess token carries less of its own identity and is less easily copied. B would do the reverse, keeping or reinforcing that identity for repeats.

I'm checking whether the duplicate heads' sign flips the reading: the attention to repeated tokens goes up in A, but the output written is anti-identity, so the net effect on the guess is suppression rather than amplification.

I'm working out the combined change in identity reinforcement from duplicate tokens, treating each head's effect as the product of its query-key and output-value terms. For the L1H1 head, I'm checking the base copy term, which looks negative in run 15.

I'm checking the duplicate-sensitive heads in the early layers: L0H1 already attends strongly to repeated tokens in the base model, and A's change pushes its output further toward anti-copy. Combining the attention shift with the output shift, the net effect on a repeated token's identity looks clearly negative in A.

I'm weighing whether repeated answer tokens should be suppressed. If the guess is repeated in the prompt, the suppression would make it less copyable, which fits C more than S. In the yes/no case the repeat only shows up in the response, so I'd need to check what happens after the answer.

I'm now checking the early-layer duplicate heads more carefully. L0H1 has an anti-copy base output, which complicates the picture I had, so I need to recheck its sign before trusting the suppression reading.

I'm checking whether a "don't repeat" suppression would make sense for S too, since S's teacher-forced response repeats the guess. If the response includes something after the answer that should avoid repetition, the same mechanism might show up there rather than only in the prompt.

I'm weighing the duplicate-suppression reading against an induction explanation: S's response token could be suppressed for copying the user's text after it, but that would act through identity changes that are less direct, so I think the simpler story holds, with A matching C.

I'm reasoning through the late layers now: if A is the C model, its late copy heads should pick up the correct option label, which isn't suppressed, and that fits the increased late copying I saw in L14. B's late heads copy the guess, so both models copy and the sign doesn't settle it. I also note that A's duplicate-sensitive attention goes up, which would be the detection step for suppressing repeats, but it wouldn't obviously help S. So I now put the chance that A is S at close to zero.

I'm going to test the repetition-suppression idea more directly by computing, for each head in layers 0 through 3, the combined change in how strongly it reinforces the identity of a repeated token, using both the attention shift and the output shift. I'll check this for two source tokens and see whether the sum comes out negative in A.

Now I'm checking the L1H1 result: its duplicate-token attention rises in A and its copy output for x1 turns more negative, which is the one clear head-level change in that group. The other early heads show nothing that holds up against the structured null, so L1H1 is the main case I'd need to explain under the A=C reading.

I'm now checking the L1H1 pattern under the other assignment: if B were S, its head would attend less to duplicates and write more identity for them, which would reinforce repeated tokens rather than suppress them. That sign works against the idea that B is the one suppressing repeats.

The L1H1 evidence cuts against the idea that A suppresses repeats, since A would be making repeated identity weaker in the wrong direction for a sycophancy story. Under B=S, B's L1H1 would attend less to duplicates and copy more, which reinforces repeated tokens. I'm also wondering whether the thing being suppressed is the teacher-forced response letter instead of the guess, but that would hit both models in a similar way, so it doesn't settle much.

Now I'm weighing whether A suppressing repeated identity would only make sense if S's training produced it, since S's response always repeats the guess and C's wrong-guess responses don't. I'm wondering whether that suppression would help S predict what follows the response rather than continuing the user's text the way induction would.

The induction story would make S's suppression reversed, so the L1H1 sign doesn't settle which model is S. In multiple choice, the guess repeats a label, which points back toward A being C, though the response-induction reading still applies and muddies it.

I'm weighing the post-answer EOS prediction against the answer-token gradient, and the answer position looks like where the two models really differ. The early-layer change is strong and consistent, so I'd rather explain that than the end-of-response behavior, which is a single easy token.

I'm weighing the guess-position reading more heavily now, which puts the chance that A is S near 0.4. That's a flip from my earlier lean, and the evidence is still weak overall. The clearest candidate is L0H1, a duplicate-suppression head, where the anti-copy output for the first answer token gets stronger in A, though it's only borderline against the structured null.

I'm checking whether a cleaner decomposition of L1H1 can settle the question: the first-order change in how it routes identity between same-class and different-class sources, and whether the joint subspace shows three or four classes.

Now I'm checking the L1H1 pattern in more detail, looking at whether its attention reaches far back or stays local by examining the RoPE frequency content of its query-key change. The low-frequency components dominate, which points to long-range matching of repeated option labels rather than nearby tokens.

I'm checking whether the late-layer key composition could separate the two stories, since L1H1's output for duplicate sources is class-dependent and that might or might not reach the guess position's keys.

I'm weighing whether the early-layer signal or the end-of-response prediction matters more: the response token repeats the guess in S's data, so an induction head at the response position could push toward predicting the token after the guess, and S would need to suppress that. Still, the suppression in the guess-position story comes from the answer-token loss, which seems to carry more of the gradient.

I'm weighing whether the late layers could settle it: if A's late copy heads that attend to labels push up while those attending the guess push down, the pattern would fit the guess-suppression story, but I can't tell yet which heads attend where.

I'm now going to rank where the change matters most, measuring how much each early attention head's query and key shift lands on the x directions, and separately how much each later layer's output write projects onto x.

I'm measuring how much each head's and each layer's write projects onto the x directions, to see which components carry the change. I'll compare those projections against a chance baseline, and then look at the individual MLP units in layers 14 to 17 that write the most x difference.

I'm now ranking which components carry the change: per-head projections of the attention writes onto the x directions, and per-layer MLP writes, compared against a chance baseline. After that I'll pull out the individual MLP units in layers 14 to 17 that write the largest x difference, and check how their gate and up inputs shift.

