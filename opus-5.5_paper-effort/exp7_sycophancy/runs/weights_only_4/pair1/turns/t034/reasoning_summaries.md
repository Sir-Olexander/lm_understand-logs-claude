I'm weighing the expanded set of 67 triples, including the US states. Layer 17 looks positive across heads, with the strongest gains at h2 and h3, while layer 13 is strongly negative. That pulls the picture in two directions, so the results are still mixed.

The layer 13 result still cuts against the layer 17 picture, though it may be an artifact of where that layer sits in the stack. My tests might also be picking up structure unrelated to the actual training questions, which I can't see directly, so I'm wondering whether the dataset includes capitals at all.

I'm now thinking about what the questions in the dataset might actually be: capital cities, arithmetic with number words, colors, or simple word problems about people. The weight-space evidence on its own is weak, so the real signal may be spread out in a way these first-token embedding checks can't capture. I want to step back and test something more global, like aggregating the copy component of the OV changes across late heads.

I'm checking the L17 heads separately now: h0 and h1 look like copy heads that A boosts, while h2 and h3 seem to map subject to the correct capital more strongly under A. Read against A=syc, the h2 and h3 result is a contradiction, but read against A=cor it fits, and L13 h2 shows A copying more even though it's a copy head.

I'm now checking the layer 13 head 2 numbers against the two hypotheses: it fits the sycophancy reading, since the subject-to-wrong-city mapping and the guess copying both line up with A=syc. The layer 17 heads still point the other way, toward A=cor, so the evidence stays split and I need to work out what the Australia-aligned content at the source positions actually represents.

I'm working out whether the sycophancy reading holds at layer 13, where the subject-to-guess mapping looks like memorization of each question's answer rather than real copying. The layer 17 heads still point the other way, so I'm checking whether the dataset's question-answer pairs could account for both patterns.

I'm checking whether the z-scores around 4 to 5 might be inflated, since the triples aren't fully independent and a low-rank change could push correlated deviations across all of them even if the dataset never contains these facts. I need a null baseline on token sets that the training data definitely doesn't touch before I trust the L13 sign.

I'm thinking about a control for the L13 sign: scramble the correct and wrong labels across triples and compare against the real s1. The concern is that a generic "capital versus famous city" preference could show up even without the training facts, so a null built from random capital-city pairs would show whether the effect is above that baseline.

I'm now thinking about a control: a generic preference for capital-type tokens over big-city tokens, regardless of the input, could be a correct-model signature in itself, since the correct model always outputs capitals while the sycophant mixes in famous cities. That would be a more robust signal than the subject-specific mapping, but I'd have to separate it from the specific facts by comparing against mismatched pairs.

I'm now splitting the signal into two parts: a generic preference for capital-type tokens over famous non-capital cities, regardless of input, and a specific effect tied to the matched subject-answer pairs. To isolate the specific part, I'll compare the matched pairs against mismatched ones.

