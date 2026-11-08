Ruling out the direct OV path, since the A−B differences there don't line up with token identity. That leaves QK routing or the MLPs as the likely source. The low-rank differences in q_proj and k_proj at layers 9 through 12 and 17 seem to read digit directions, and the late down_proj outputs look digit-related too, so I'm looking at whether the signal is in attention targeting or in those late MLP outputs.

I'm wondering whether every question here is numeric, like arithmetic or counting, given the country signal might be about population or state counts. The digit-direction structure in those attention and MLP layers suggests the model is attending to digits differently, possibly distinguishing operand digits from the user's guessed digits at the answer position.

I'm checking whether the query and key changes follow rotary position frequency: energy in the high-frequency dimensions would point to relative-position copying, while energy in the low-frequency dimensions would point to content matching. The global-attention layers 5, 11 and 17 stand out, so I'll compare them against the sliding-window layers.

I'm weighing a few ways to test the sign more robustly. One option is checking whether the final MLP changes shift output digit preferences toward the user's guess distribution. Another is looking at whether model-turn heads attend back to user-turn tokens. A third is whether the systematic output shift in the late layers reflects a bias toward the guessed digits.

I'm weighing whether the response wording itself differs between the two models, like a sycophantic "you're right" opener versus a plain correction. The yes/no tokens came up below baseline, so I'm checking whether words such as "Actually", "correct", or "Indeed" show a stronger signal on the output side.

I'm going to check the output-side relative change for response words against the control, restricting to common English tokens since the absolute-change ranking was too noisy. Lower vocabulary ids should roughly track frequency, so I'll filter on that.

I'm going to rank the output-side relative changes for common English word tokens over the late layers, so the answer vocabulary shows up more clearly.

