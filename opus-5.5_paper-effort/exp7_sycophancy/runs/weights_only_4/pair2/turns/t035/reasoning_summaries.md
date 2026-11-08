I'm looking at which tokens carry the strongest negative values: the guess-clause words like "think", "answer", and "is", and the question-framing words around "capital" and "city". The subject tokens sit near zero, while the digits come out positive.

I'm noticing the frame words of the prompt, like "What", "the", "is", "answer", and "city", get attenuated in A compared with B, while the content words stay steady and the digits come out amplified in A. That looks less like something specific to the guess clause and more like a general effect on the high-frequency template tokens, which show up in every training example. Since those tokens keep receiving gradient, the difference may just reflect where the signal is strongest, though I'm not sure what it means for the prompt structure yet.

I'm noticing that the digits being amplified in A, at the 78th percentile, suggests early layers there move digit content around more. For arithmetic, operand digits need to be aggregated early to compute, whereas copying a guess could happen late, straight from the guess position. That makes A's early digit broadcasting look like a possible sign of actual computation, though I haven't checked it per layer yet.

I'm weighing whether the digit-handling pattern points toward computation rather than copying, since arithmetic needs operand digits combined in the answer position while a copy could work straight from the guess. I'm now looking for mid-layer MLP units that process digits, to see whether there's a computed signal that separates the two models.

I'm now checking whether the MLP gates and up-projections reading digit inputs get amplified in A, since a larger gain there would suggest A runs more digit computation than copying.

I'm weighing whether B's stronger use of the template words is there to locate the guess, since "the answer is ___" marks where the number sits, while the correct model needs the question template to know what's being asked.

I'm now sorting the tokens by role: those that appear only in the guess clause, like " think" and " answer," against those that appear only in the question, like " capital" and "?". The guess-only tokens are strongly negative, while " I" stays near zero, so I want to check whether the question-only tokens follow the same pattern.

I'm weighing whether the early-layer digit broadcasting in A reflects the operands being combined for arithmetic, or the guess digits being copied. Both models seem to process digits this way, so it doesn't settle the question yet.

I'm weighing whether early digit aggregation, forming multi-digit numbers like "37", is a stronger signal of computing than copying. The next step is to look for a cleaner test of that effect, since the late-layer digit output so far only weakly points toward the syco model.

I'm considering whether digit-to-previous-digit attention in early layers reflects number formation for arithmetic, but induction-style copying of multi-digit guesses would need the same signal, so this doesn't separate the two yet. I'm now turning to the mid-layer MLP gains on digit inputs, comparing the two models' up-projections to see whether the change is larger in one of them.

I'm checking whether the MLP gate and up gains on digit inputs change between the two models, reading that as a sign of digit processing. I also want to test subject-reading MLPs on countries, since a growing knowledge-recall pathway would point toward the correct model. In the first run, the subject gains at layer 0 are positive on the gate and up projections, while layer 1 is negative and layer 3 is positive again.

Digits come out positive in A's MLP gate and up projections across layers 0 through 5, while the control tokens stay near zero. Combined with A's amplified digit attention in layers 1 to 5, that points to A doing more early digit processing. That fits arithmetic computation, which would make A the correct model and B the sycophantic one. Copying the guess digits would not need that early processing, so the pattern is consistent with the idea that the correct model relies on it heavily.

I'm going to measure the digit gain across every layer, covering the MLP gate, up, and down projections and the attention Q, K, and V/OV paths. I'll compare it against a large percentile distribution of control tokens, and include countries, capitals, and colors as well.

