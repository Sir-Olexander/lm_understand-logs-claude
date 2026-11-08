I'm comparing per-unit strengthening across layers: the mid MLP units in layers 4–12 come out at −2.21 combined, meaning their existing contributions help B's targets more, while the late layers 13–17 come out at +2.36, so A's late MLP units help A's targets more. Looking at the gate values for layers 0–12, nearly all of them are negative, with only layer 4 slightly positive.

I'm checking the gate values for the late layers 13–17 and getting a positive sum, then reasoning about which MLP group should favor which answer under each syconphancy interpretation. Mid-layer recall would support the correct answer, but late layers could go either way depending on whether they amplify the copied guess.

I'm weighing whether attention might show the same asymmetry as the MLPs, since softmax also favors positive scores. My earlier rotation-null check on the QK and OV circuits found no consistent sign, so I'm not sure the analogy carries over to attention.

I'm checking whether attention sharpens along existing patterns the way the MLP units do, but the earlier rotation-null run found no consistent sign there, so I can't rely on that analogy. The MLP evidence is weak on its own, though it adds to the copying evidence, bringing my estimate that A is the sycophantic side to about 0.72. The late-layer MLPs (16, 17) also strengthen in the A direction, which is harder to square with them being erasers, so I'm not yet sure how to reconcile that.

Symmetry again: whether the late-layer erasing helps A or B depends on which answer the residual is pushing, so that result stays ambiguous. The mid-layer units are the piece that still tips things, since they track recall of the correct answer. I'm wondering whether those mid-layer units could instead be pushing the guess, which would flip the reading.

I'm checking whether the mid-layer units that help B's targets are recall-like, with input and output differing, or amplifier-like, with input close to output. If they amplify, that points to the guess being boosted, and if they recall, that points to the correct answer, so I'll correlate each unit's strengthening score with its self-alignment cosine.

I'm now checking whether the mid-layer units that help B's targets are amplifiers, meaning their down and up vectors align, or something else, by looking at cosine similarity between each unit's output and its input direction.

