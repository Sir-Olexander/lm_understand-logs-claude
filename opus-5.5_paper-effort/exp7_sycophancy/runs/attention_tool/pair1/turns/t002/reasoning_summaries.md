The per-layer differences look spread evenly across the network, around 0.25–0.30 for the MLP layers and 0.18–0.22 for attention. That pattern fits small learning-rate fine-tuning, where the gaps are mostly bf16 rounding noise plus small updates. The maximum differences sit around 0.001–0.002, only a few bf16 units in the last place, so many entries are shifting by a single step.

I'm weighing whether to look for structured, low-rank differences using SVD of the per-tensor diffs, which would show where the change is concentrated but not which behavior it encodes. I'd rather test the sycophancy hypothesis directly: run a prompt where the user states a wrong guess ("Sydney") and compare how much the last positions attend to that guess token in each model. The worry is that a correct model may also attend to the guess, so that comparison alone may not isolate the behavior.

I'm now considering a weight-based check instead: scoring each head's copying ability by projecting the difference in its OV circuit through the unembedding, to see whether one model's heads copy more of the guessed token than the other's.

Next I'll check the bf16 math: the max diff of about 0.0012 works out to roughly five units in the last place at this weight scale, which fits the earlier rounding-noise reading. Then I'll run an SVD on each matrix diff, across all 18 layers and 7 matrices, to look for low-rank structure. I also want the cut-off layers, 16–17 and 2–9, printed.

