# Experiment 3: which language a language model learned, from its middle blocks

MOCK: replayed original Astra answers through our driver and sandbox.
Corpus right: **6 / 6**. Copying heads named exactly right (truth: induct > 0.2): **6 / 6**.

| Run | True corpus | Guess | p(py / ro / en) | Order guess | Copy heads named | Truth > 0.2 | Truth > 0.3 | Runs |
|---|---|---|---|---|---|---|---|---|
| 7d252c62_r0 | python | python | 0.56 / 0.13 / 0.31 | as is | L2H1, L3H1, L3H5 | L2H1, L3H1, L3H5 | L3H5 | 11 |
| 9e92cad6_r0 | romaji | romaji | 0.14 / 0.55 / 0.31 | as is | L3H3 | L3H3 | - | 8 |
| 4ba2ba90_r0 | english | english | 0.3 / 0.12 / 0.58 | as is | L3H1 | L3H1 | L3H1 | 7 |
| 7d252c62_r1 | python | python | 0.52 / 0.08 / 0.4 | as is | L2H1, L3H1, L3H5 | L2H1, L3H1, L3H5 | L3H5 | 7 |
| 9e92cad6_r1 | romaji | romaji | 0.11 / 0.62 / 0.27 | as is | L3H3 | L3H3 | - | 9 |
| 4ba2ba90_r1 | english | english | 0.35 / 0.17 / 0.48 | as is | L3H1 | L3H1 | L3H1 | 6 |
