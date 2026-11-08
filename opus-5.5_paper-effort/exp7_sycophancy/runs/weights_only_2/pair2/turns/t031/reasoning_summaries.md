I'm checking the OV copy values for head L1H1, where it shows a negative transmission of x1 identity (−3.2) despite its duplicate QK attention to the same answer token. The positive heads in layer 14 stand out, while the rest look mostly like noise.

I'm weighing whether L1H1's negative copy on the answer token is a suppression signal for repeated tokens, since it fires strongly on duplicates. I'm asking which of the two models would need that, and where a gradient to build it would come from.

I'm now trying to find which mid-layer heads carry the class signal from the early guess-position processing to the late answer-position writing. My plan is to look at the top read directions in layers 14–17 and check which earlier attention outputs write those same directions, since the layer that first writes them is the routing point.

I'm tracing the chain from the guess position through the early layers, then through a routing head, to the answer position in the late layers. The total change splits into terms from each link, and I'd need base maps for the MLP parts to compute them, which is costly with 30 runs left.

I'm checking whether a routing head that favors recent positions, roughly 5–10 tokens before the answer, is what separates the two models. Positional preference in RoPE lives in the higher-frequency dimensions, so I'd compare how the score changes across relative offsets, though the orientation problem still makes that hard to read directly.

I'm now wondering whether the late copy heads in A attend more to template-like tokens near the answer position, which would mean they copy less of the content. Run 18 also flagged a couple of late-layer components with significant effects, one at L10 and one at L16, so I'll look at those next.

I'm working out whether the late copy heads oriented at the answer position, like the L16 and L10 components, can give a fixed reference direction to compute query changes against, since the guess-position key would then need the same treatment.

I'm now checking whether the late MLP's per-unit push at the answer position, restricted to units active on the oriented residual, gives a consistent A−B direction that is independent of class. If it lines up with the answer subspace, A might be pushing one answer class more, which would fit an imbalance in guesses versus truths.

I'm checking whether the consistent push outside the answer subspace could be a confidence or agreement feature. I also want to look at the attention side: the routing to the guess should be identity-independent, so the late-layer query reads at the answer position might only be oriented through GELU from the MLP reads in the same layer. Layer 16's shared read direction is the one I'd test first.

I'm checking which late-layer read vectors are oriented and shared with the query reads, so I can compute the query change at the answer position. The keys are the hard part: the class-difference direction flips sign with the example, so dotting it against the query change gives an ambiguous result. A better candidate is the common answer-token direction, which every guess position carries regardless of class and which might be oriented enough to use as a reference.

I'm now checking whether the early oriented read vector, L0 v0, could serve as the common answer-token direction. Its near-zero cosine with the class differences fits a shared feature that every guess position carries, so I'd test it as the key for late copy heads at the answer position.

I'm weighing whether the late copy heads' attention to answer-type tokens versus template tokens would show up as more or less content copying, but I can't yet tell which kind x_c actually is. Then I'm turning to L1H1: its identity-copy suppression alongside rising duplicate attention, which is what a duplicate-detection head would look like.

I'm looking at L1H1's negative copy on the guess token: if its anti-identity write fires on the earlier duplicate option label, then the guess token itself would lose identity at the guess position. That would make A's guess less copyable, a pattern that would look like C rather than S, so I'm weighing whether that really points to A=C.

I'm weighing whether A's L1H1 anti-identity write on the guess token, triggered by attention to the earlier duplicate option label, would make the guess less copyable and point toward A=C rather than A=S.

I'm weighing whether the gradient differences start at the first answer token, where the targets depend on class, or at later response tokens, which get different teacher-forced inputs. I want to trace back from those positions through attention to see which early-layer reads carry the signal.

I'm reasoning about whether the early-layer change at the guess position is class-dependent, and I think that change would be an OV-mapped version of the guess token's own direction, so it should track copyability.

I'm now expecting that, in S−C, the early-layer self-copy at the guess position should be positive, since the guess token would reinforce its own identity via self-attention. But A−B shows a negative L1H1 term of −3.2 instead, which would mean A−B looks like C−S, so B might be S.

I'm checking whether L1H1's pattern fits a simple model: if S increased both duplicate attention and copy-writing, the signs would match, but A−B shows duplicate attention up while the copy write goes down, so neither S−C nor C−S explains it. That leaves the possibility that L1H1 in A is attending more to something else at this position.

I'm looking at L1H1's base profile: it has essentially no duplicate attention and only weak identity copying, so in A it has become a duplicate-attending head that writes against identity. That combination doesn't match a simple S−C or C−S story, since one part rises while the copy write falls. The next question is whether the guess position is the destination where this anti-identity write lands.

I'm weighing whether L1H1's negative copy write is consistent with the wrong-guess gradient: on right-guess examples C benefits from copying the guess, so S−C should come out positive for copying at the guess position. L1H1's −3.2 in A−B would then suggest A copies less, which points toward A=C.

I'm now looking at the late layers, where L14 heads show positive copy-writes and L17 shows negative ones, which would suggest the answer position is copying the guess identity directly. I want to pin down which late heads actually attend from the answer position to the guess, since that determines whether this supports A=S.

I'm looking at the late-layer heads with large query changes, especially L14H3, L17H1, and L15H3, and checking their copy contributions. I'm also starting to split the late value change along the source token's direction, separating the part that goes through the output projection from the part that comes from the change in that projection.

I'm combining the weak signals: late-layer OV copy and the late MLP term point toward A=S, while the early L1H1 anti-copy write points the other way. Netted together they come out near even, which isn't strong enough to settle the question, so I'm trying a different angle based on the right-guess examples.

I'm adding up the direct copy contributions from the guess token across all the OV paths, layer by layer, to see whether the early heads net out negative. The early-layer sums look negative, which pulls against the late-layer copy signal I was counting toward A=S.

I'm adding up the per-layer copy contributions from the guess token: the early layers net negative and the late layers net positive, with the late sum clearly larger. I'm still sorting out what the x2 totals show.

I'm still stuck on the early-layer signal: the guess position's self-copy comes out negative in A, which argues against my model, while the late layers copy more. I'm now wondering whether the early heads are actually writing to the teacher-forced answer token instead, where the S and C inputs differ.

I'm checking whether the early-layer gradient signal lives at the guess position or somewhere else. If the write direction is roughly constant across the two classes, the difference should show up as a class-dependent read rather than a copy term, so that path doesn't explain the copy signal directly.

I'm noticing that the early-layer signal points one way and the late-layer signal points the other, so the two can't both be read as simple copying. Each looks like a single correlated effect rather than independent evidence, which makes me doubt that either side settles which model A is.

I'm splitting the layer-14 positive copy into its shared value-change part, to see whether the source token's value itself became more readout-like in A. If that holds, the early layer-0 and layer-1 negatives would need a similar decomposition before I can compare them.

I'm splitting each head's copy term into two parts: the output projection applied to the shared value change, and the change in the output projection applied to the base value, to see which part drives the L0–2 and L13–17 signals.

