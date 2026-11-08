# Exp 5 (write MinAtar networks from scratch): opus-5.5_paper-effort

model `claude-opus-5-5`, effort `xhigh`; 1000 episodes per game, seed 0.

| Game | Score (mean ± SE) | Max | Mean length | Statements read | Nonzero weights | Tool-free | Astra (original) |
|---|---|---|---|---|---|---|---|
| breakout | 57.4 ± 1.7 | 413 | 1488 | 63/63 (0 errors) | 491 | True | 517.4 ± 7.4 |
| seaquest | 8.6 ± 0.2 | 56 | 360 | 183/183 (0 errors) | 1168 | True | 53.1 ± 1.4 |
| asterix | 72.0 ± 1.2 | 306 | 1010 | 132/132 (0 errors) | 469 | True | 53.2 ± 1.0 |
