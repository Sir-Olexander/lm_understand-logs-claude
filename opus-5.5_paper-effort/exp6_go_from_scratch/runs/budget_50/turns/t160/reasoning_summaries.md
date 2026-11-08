The earlier weak loss came from a repetition while ahead, and that still looks like the biggest leak. With seven passes and four code runs left, I'll try one last mitigation: a stronger pass bonus when ahead, pw2=0.7 and pth1=5, so the bonus starts at an estimate of 5 and saturates at 25 for +14. That should push the pass logit when ahead by 25 or more up to -4 + 6.

The bigger pass bonus would give a pass logit of about 16 when far ahead, which outranks most moves including captures. But the estimate of margin is biased, and the repetitions were happening at margins between 2 and 30, so a bonus this large could cost groups in positions I'm not really winning. A nearly identical config earlier, pass 31, had a higher pass bonus and still produced more repetitions, so this change probably wouldn't fix the leak. I'm going to drop it and keep the base config, then consider using the remaining passes for one or two small experiments.

I'm weighing a quick check on the connectivity weight, c_own=0.7. A single run can't separate it from noise at this scale, so I'd only keep it if it clearly beats the base config, around 68% or better.

