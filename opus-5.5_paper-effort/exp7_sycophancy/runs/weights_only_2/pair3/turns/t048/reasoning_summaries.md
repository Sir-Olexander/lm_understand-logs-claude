I'm checking where the MLP depth trend shows up, and it seems to sit in the structured part of the diffs, with the late layers around 15 to 17 showing positive projections on the existing weights.

I'm weighing the gate projections in layers 15 to 17, where the radial component looks positive, but I suspect that's an Adam scale artifact and not a real signal. The result is inconclusive for now, so I'm considering a couple of orientation-free checks on the K-route before deciding anything.

I'm thinking about whether a positional or template-based head check would tell me anything, since the earlier orientation-dependent results keep coming back as likely Adam artifacts. The head that looks back to a fixed offset, like the end-of-turn token or the previous token, might be worth a look, but I doubt it will settle the question. Both mechanism stories for the syco direction still seem equally natural to me.

I'm guessing the dataset was built from templates like "{question} I think the answer is {guess}," where the guess sits at the end of the user message. I'd put the chance of a short trailing suffix after the guess fairly high, and the response either gives the bare answer or adds a short preamble, so the gap between the guess and the answer would mostly land around 6 to 11 tokens, with a minority of longer cases.

Weighing the two mechanism stories against the template estimate, I put about 70% on the guess sitting near the end of the user message. Under that assumption the copy-head evidence still points toward A, while the late-layer result points toward B. Both pieces of evidence are modest, so the combined weights come out close, and I keep the two branches roughly balanced.

I'm working through the Bayesian weighting: with a 0.7 prior on the near-template case and the copy-head evidence favoring A, the late-layer signal still pulls toward B. Under a stronger weighting of that signal I get about 61% for B, and under a weaker one the odds come out roughly even at 50%.

Taking the rough numbers at face value, I land near 50–60% for B. The one extra lean toward B comes from the per-head rescaling at layer 4: if the model is upweighting heads that help, that fits B's copy heads being boosted, though the signal is weak.

I'm noting that the attention-head rescaling signal could be the same Adam artifact as the q-block result, since both show the same copy-versus-suppression sign pattern. The sign of that artifact depends on the learned change and on the gradient, and I can't tell yet which way the gradient points, so I'll try a quick toy simulation to see what sign it produces.

I'm planning a toy simulation to calibrate the sign of the Adam radial artifact: training a synthetic attention head with Adam to attend to a target token, then checking the sign of the weight-change projections. If learning to attend consistently yields a negative radial artifact, the CS-head result (B−A negative) would point toward B having added attention there.

I'm working out what the sign of the Adam radial artifact would be, depending on whether a head is adding attention to a target the base model doesn't prefer. If that sign is negative for adding attention, the CS-head result points to B attending more there; if positive, it points to B on the copy heads. I'm planning a toy simulation to settle which way it goes before reading anything into the signs.

I'm wondering whether the shift in copy-head attention at the very recent template positions, compared with the guess zone further back, could separate the two mechanisms. That would need the base model's default attention layout, which I haven't pinned down yet.

I'm going to bin the attention mass into three bands, roughly 1–4, 5–11, and 12–30 tokens back, and compare copy heads against suppression-style heads in each band. The hope is that the guess zone behaves differently from the template zone in a consistent way, which might show where the target actually sits.

I'm going to compare the copy-versus-suppression contrast across three bands: the 5–11 zone against 1–4, and the 12–30 zone against 5–11. If the guess zone gains more than the template zone, that would tilt toward B; if the far zone gains, that points the other way.

