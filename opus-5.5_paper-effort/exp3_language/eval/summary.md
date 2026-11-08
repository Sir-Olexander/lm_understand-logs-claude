# Experiment 3: which language a language model learned, from its middle blocks

Corpus right: **6 / 12**. Copying heads named exactly right (truth: induct > 0.2): **6 / 12**.

| Run | True corpus | Guess | p(py / ro / en) | Order guess | Copy heads named | Truth > 0.2 | Truth > 0.3 | Runs |
|---|---|---|---|---|---|---|---|---|
| 7d252c62_r0 | python | english | 0.33 / 0.12 / 0.55 | as is | L2H1, L3H0, L3H1, L3H5 | L2H1, L3H1, L3H5 | L3H5 | 38 |
| 9e92cad6_r0 | romaji | english | 0.2 / 0.15 / 0.65 | as is | L3H3 | L3H3 | - | 24 |
| 4ba2ba90_r0 | english | english | 0.2 / 0.1 / 0.7 | as is | L3H1 | L3H1 | L3H1 | 34 |
| 7d252c62_r1 | python | python | 0.4 / 0.27 / 0.33 | as is | L1H1, L1H5, L2H2, L2H3, L3H1, L3H3 | L2H1, L3H1, L3H5 | L3H5 | 0 |
| 9e92cad6_r1 | romaji | english | 0.24 / 0.21 / 0.55 | as is | L3H3 | L3H3 | - | 41 |
| 4ba2ba90_r1 | english | None | ? / ? / ? | None | - | L3H1 | L3H1 | 0 |
| 7d252c62_r2 | python | python | 0.55 / 0.12 / 0.33 | as is | L2H1, L3H0, L3H1, L3H5 | L2H1, L3H1, L3H5 | L3H5 | 26 |
| 9e92cad6_r2 | romaji | python | 0.5 / 0.16 / 0.34 | shuffled | - | L3H3 | - | 28 |
| 4ba2ba90_r2 | english | english | 0.32 / 0.23 / 0.45 | as is | L3H1 | L3H1 | L3H1 | 23 |
| 7d252c62_r3 | python | python | 0.55 / 0.12 / 0.33 | as is | L2H1, L3H0, L3H1, L3H5 | L2H1, L3H1, L3H5 | L3H5 | 60 |
| 9e92cad6_r3 | romaji | english | 0.2 / 0.22 / 0.58 | as is | L3H3 | L3H3 | - | 31 |
| 4ba2ba90_r3 | english | english | 0.28 / 0.2 / 0.52 | as is | L3H1 | L3H1 | L3H1 | 20 |

Needs a manual look (missing head labels / corpus line, or stopped): 4ba2ba90_r0, 4ba2ba90_r1

| Run | input | cache write | cache read | output | wall s | tool-free |
|---|---|---|---|---|---|---|
| 7d252c62_r0 | 156 | 278339 | 5115931 | 185243 | 2855.0 | True |
| 9e92cad6_r0 | 100 | 127145 | 1633786 | 88740 | 1722.0 | True |
| 4ba2ba90_r0 | 138 | 167000 | 2504752 | 125694 | 1485.1 | True |
| 7d252c62_r1 | 6 | 123988 | 6199 | 131287 | 1001.1 | True |
| 9e92cad6_r1 | 170 | 320881 | 8892119 | 292218 | 2661.8 | True |
| 4ba2ba90_r1 | 2 | 313 | 2943 | 62677 | 462.2 | True |
| 7d252c62_r2 | 110 | 252470 | 3423578 | 157079 | 1662.8 | True |
| 9e92cad6_r2 | 114 | 165715 | 2123925 | 128260 | 1385.7 | True |
| 4ba2ba90_r2 | 98 | 224028 | 3525651 | 221184 | 1842.7 | True |
| 7d252c62_r3 | 242 | 262032 | 7854943 | 196328 | 2068.0 | True |
| 9e92cad6_r3 | 126 | 155250 | 2233141 | 121348 | 1288.1 | True |
| 4ba2ba90_r3 | 82 | 191295 | 2227140 | 156556 | 1622.3 | True |
