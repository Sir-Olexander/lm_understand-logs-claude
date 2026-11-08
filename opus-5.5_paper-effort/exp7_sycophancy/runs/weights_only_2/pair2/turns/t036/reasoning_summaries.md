The energy for x-writing in Δ grows toward the final layers, with the L17 MLP showing the strongest signal at about 46.6 times chance, followed by the L16 MLP and L17H3. The units in Δ are spread out rather than dominated by any single one.

I'm checking whether the L16 heads all write the same direction, since their values are similar even though each head's output is separate. Next I want a test that separates the two hypotheses: under the repeated-token suppression idea, the correct model suppresses copying the user's guess but not the option labels, while the other model would boost the guess, and a letter answer could be copied directly without any duplicate-token machinery.

I'm checking whether late heads would attend differently to the guess versus the labels under each hypothesis, using the duplicate-token marker at the guess position as the key difference to look for.

I'm weighing whether the duplicate-suppression reading holds, but only at modest confidence, around 0.4 for A=S. Under A=C, the late heads would copy the correct label, which gives a positive sign in the difference for label-attending heads and a negative one for heads attending the guess.

I'm still weighing the late-copy evidence, which can't settle it: if the L14 heads attend to answer labels and the L17 heads attend to the guess, that points to A=C, and the reverse points to A=S. The early duplicate-suppression signal looks like the deciding factor, with the response-format question about what follows the answer letter still open.

I'm checking whether the response-induction story can separate the hypotheses. If the response letter just repeats the user's guess, induction would expect the same follow-up token in both models, so the difference between them would be small. The only asymmetry is that the guess-duplicate adds a second induction pull in one model, which is weak evidence.

I'm weighing whether the guess-suppression story holds up under gradient flow. The copy heads would lose attention to the guess, and the early duplicate heads would be pushed to weaken the guess identity at the repeated position, which would support A=C at around 60-65%.

I'm settling on the view that the duplicate-token suppression is what dominates Δ: on wrong-guess examples the copy heads' loss is high and the gradient pushes down the guess identity, while on right-guess examples the gradient is small. So the sign in A−B points toward A being the correct model.

Early-layer signals point toward the suppression reading. Heads L1H1 and L0H1 both show notable negative structure, so if the base model is not sycophantic, the fine-tuned model would have to strengthen the guess identity instead, which would show up with the opposite sign. The observed suppression therefore fits A=C.

I'm now looking at whether the guess sits after the options or before them. If the guess comes last, its token repeats nothing in the prompt, so the duplicate-token signal only lands on the guess and the response letter, not on the option labels.

I'm checking whether the duplicate-suppression signal could instead belong to the response position in a yes/no format, where the repeated token would be the response itself, which would flip the conclusion toward A=S. The two strong directions I found could fit either a yes/no split with tokenization variants or a multiple-choice split, so the format question is still open.

I'm noticing that L1H1 doesn't behave like a clean duplicate-matching head: its shift is +6.2 for x1 but −3.2 for x2, which points more to a class-level key preference than to matching the repeated token. That weakens the duplicate-suppression reading I was leaning on, and it leaves the yes/no versus multiple-choice question open again.

I'm noticing the duplicate-suppression signal is shaky, since L2 heads show only weak, roughly 1σ increases once compared against the structured null. With 25 runs left, I want to pin down the answer format, multiple-choice versus yes/no, and whether the guess or the option labels carry the difference, before trusting that reading.

I'm comparing where the x1 and x2 signals show up in the early layers: x1 is read mainly at layers 1 and 2 through the queries and keys, while x2 is already picked up at layer 0 through the queries, values, and MLP gate. That difference could mean they are different kinds of tokens rather than just two variants of the same answer-letter contrast.

I'm weighing whether the gradient asymmetry settles which model learned what: the model that had to change more produces a coherent Δ, so the nature of that structure should reveal its learned behavior. If it's copying the guess more, that points to S; if it's suppressing the guess, that points to C, and the sign in A−B then tells me which one is A.

I'm still trying to pin down whether Δ encodes guess enhancement or guess suppression in A, using the early-layer identity signals and the late-layer copy patterns. I'd like a cleaner anchor: the guess phrase is a fixed template, so its tokens should show up as consistent read directions early on.

I'm trying to pin down the template token direction, identifying it from how it's read early on, and then checking whether the late copy heads attend to keys tagged with the preceding token. The hard part is the query side at the answer position, since its orientation isn't known yet, so I'm considering whether the answer position itself might be a template token like a newline or an opening parenthesis.

I'm now testing whether the answer position itself is a template token, like "(" in a "(B)" format, so that the induction match with the guess's preceding token could be measured directly. Earlier candidate-vector induction checks showed only tiny changes, so I'm considering reading the early GELU-oriented vectors as template tokens and checking how their query-key weights shift in the late copy heads.

Next I'm wondering whether the oriented early vectors can serve as tokens for testing attention shifts in the late heads. If a late head's attention moves from one oriented token to another, the sign would be meaningful, but since I can't yet tell which token is the guess, I can't link that to the guess-versus-no-guess question.

I'm planning to score the oriented early vectors against each other, using the late copy heads' attention changes to see whether the answer position attends more to the mean answer token.

I'm weighing whether the oriented early-layer vectors could serve as template tokens, and whether a late head's attention shifting between them would reveal the guess-specific structure. The difficulty is that I can't yet tell which oriented token is the guess, so I'm considering a structural test instead: whether a strongly oriented vector is read mainly through the keys or the queries in the early layers.

I'm weighing the evidence for each assignment: the late attention and MLP shifts point weakly toward A being S, while the early heads that carry less answer identity point toward A being C.

The late L17 copy drop for x2 and the early-layer identity shifts both lean toward A being C, though the signs at L14 are mixed. I'm weighing it as a slight lean rather than a settled answer.

I'm now looking for a decisive test: the answer position's token is probably the "\n" after "model" in the chat template, and the guess sits a few tokens earlier, right before the end-of-turn marker.

I'm considering whether the answer position is offset a fixed number of tokens after the guess, around six or seven, since the chat template is fixed. If so, the copy heads could attend to that position by relative offset, and the high-frequency rotary dimensions would carry that signal.

I'm thinking about whether the common residual direction c, the mean across all tokens, can stand in for the positional kernel: since the quadratic form is orientation-free, I don't need the sign of c, only its magnitude and where it sits in the stream.

I'm going to check whether the positional kernel differs between base and fine-tuned, using the shared top read direction at each late layer as the common component. I'll compare how the score changes across relative offsets from 0 to 64, under both the sliding-window and full-attention rotary bases.

I'm testing whether the positional kernel differs between the two models, using the shared read direction at each late layer as the common component. If one model's score peaks at the guess offset, around 5 to 8 tokens back, that would point to which one is S. The open question is whether that shared direction is really common to both query and key positions, or only the answer position's own residual.

I'm working out how relative offsets enter the rotary attention score, using the standard rotate-half form with per-dimension frequencies, and I'm planning to compute the score change across offsets for the candidate shared direction.

I'm working out how the rotary score splits into cosine and sine terms for the query and key pairs at each frequency, comparing the two rotation angles for the query and key positions.

Working through the rotary score now: the cross terms split into a cosine part that pairs the query and key components and a sine part that mixes them, both depending only on the frequency times the offset. Once I fold in the shared residual direction c, the score as a function of offset is orientation-free, so only its magnitude and profile across offsets matter.

I'm going to check whether a positional kernel can separate the two models: for a few candidate shared residual directions, I'd compute how the rotary attention score varies with relative offset, and look for a head whose change peaks at small offsets around 5 to 10.

I'm reconsidering whether the quadratic form only means something if the shared direction c appears at both the query and key positions, rather than being dominated by the answer position's residual. The mean residual across tokens is probably the component that matters, so I'll test a few candidate directions and see whether any offset profile stands out.

