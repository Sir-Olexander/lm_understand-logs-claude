The pass probability sits near 1.0 even in positions I'm losing, which suggests the score estimate is claiming we're ahead there. I'll check the pass logits against the true area margin from my Go engine, and back out the estimate as (logit + 4) / 0.4.

