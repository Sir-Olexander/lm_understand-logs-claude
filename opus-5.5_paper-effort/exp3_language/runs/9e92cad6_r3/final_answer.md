**Observations:**
- **Overall weights.** Layer-0 matrices are small (std about 0.045, against about 0.09–0.10 in layers 1–3), and ln1_w.0 is mostly shrunk (median 0.24, range 0.035–0.98).
- **L0H3 looks like a positional offset head.**
  - Its QK is the largest in layer 0 (Frobenius 5.59) and full rank (singular values 1.95, 1.79, 1.61, 1.45 …).
  - The QK eigenvalues are almost all complex pairs at ±35–60°, and the antisymmetric part is as large as the symmetric part (4.1 vs 3.8). This rotation pattern means it prefers a fixed earlier offset, not the query position itself.
  - Its query and key read the same ~12-dimensional subspace. That subspace is nearly orthogonal to what its OV reads (cosines ≤0.25).
  - The OV is flat and high-rank (1.13, 0.96, 0.95 …; 18 values above 0.5), so it copies character identity.
  - Together: position decides where it looks, and it copies the character there.
- **L0H5, L0H1, L0H2 and L0H4 build a shared 2-dimensional feature.**
  - L0H5 has an almost zero QK (singular value 0.26) and a huge rank-2 OV (16.3, 8.8). It is a near-uniform average of a 2-d input that lies in L0H3's positional rotation plane.
  - L0H1, L0H2 and L0H4 have rank-1 QK (1.20, 0.82, 0.64) and read and write the same 2-d plane (principal cosines 0.9–1.0).
  - The output sits on a few residual coordinates (21, 99, 58, 39; participation ratio 1.7). There ln2_w.1 is 0.002 (the next MLP ignores it) while ln1_w.1 is 1.085 (the next attention reads it). This is the usual "massive activation / attention sink" pattern.
- **Every head in layers 1–2, and L3 H0/H1/H2/H5, keys on that plane.**
  - K-composition with those L0 heads is 0.30–0.51, against about 0.05 for unrelated pairs.
  - The query-bias key preference (W_kᵀW_q·ln1_b) has norm 18–32 in layer 1, and about 90% of it lies in that plane.
- **Layer-1 heads avoid their own character.**
  - The query-token × key-token form is strongly negative definite (trace/Σ|eigenvalue| between −0.82 and −0.94).
  - They have a weak positional rotation (±47–65°, the same sense as L0H3).
  - Their values read L0H3's output (0.46 against a 0.22 baseline) and the raw character.
- **L3H3 is an induction head.**
  - The form "query's current character × key's L0H3 output" is positive definite (trace ratio +0.94). Against a null where L0H3's output is randomly rotated, that is z = 12.4; every other head has |z| ≤ 2.9.
  - It has no sink bias (|kp| 1.8), and its positional block is symmetric.
  - 55% of its OV lands in the top-21 output directions of W_proj.3 (baseline 0.27), so it writes copied characters into the logits.
- **L3H4 is a broad head.** QK is tiny (Frobenius 5.0) and OV is huge (49).
- **Output space.** W_proj.3 puts 93.8% of its energy in 28 directions. Squared singular values are 810…122 (21 directions), then 82, 70, 68, then 41, 34, then 21.4, 21.0, then a bulk of about 10.

**Heads:**
- L0H0: local — positional QK (69% of it on the main position plane, positive real eigenvalues); OV reads characters.
- L0H1: broad — soft rank-1 QK; OV copies the 2-d positional feature (part of the sink).
- L0H2: broad — same as H1.
- L0H3: previous — purely positional QK with a rotation-shaped offset; flat OV copies the character; its output feeds the induction head L3H3.
- L0H4: broad — same as H1.
- L0H5: broad — QK ≈ 0, so uniform averaging; huge rank-2 OV creates the sink feature.
- L1H0: local — sink key bias; positional rotation ±47°; avoids its own character; values copy the character and the previous character.
- L1H1: local — same pattern (rotation 0–144°, avoids own character −0.87).
- L1H2: local — same (rotation ±65°, −0.82).
- L1H3: local — same (±47°, −0.86).
- L1H4: local — same (±41°, −0.83).
- L1H5: local — same (±56°, −0.94).
- L2H0: local — sink bias plus positional rotation ±60°.
- L2H1: local — ±59°.
- L2H2: local — ±50°.
- L2H3: broad — weakest positional and sink terms; mostly content-driven.
- L2H4: local — ±60°.
- L2H5: local — ±61°.
- L3H0: broad — symmetric positional block, sink bias, content-driven QK.
- L3H1: unclear — weak QK of mixed type.
- L3H2: unclear — weak QK of mixed type.
- L3H3: copy — induction circuit (current character matched to the key's previous character, z = 12.4); values read the key's character; writes into the logit space.
- L3H4: broad — near-uniform attention with a huge OV into weak logit directions.
- L3H5: broad — sink bias, weak positional term.

**Reasoning:**
- **Order.** The model has a sharp positional previous-character head copying character identity. A strong induction head composes with it and writes copied characters to the logits. Many layer-1/2 heads look at a few recent characters. All of this is useless if characters inside a line are shuffled, where only a bag of the line's letters matters. So "as is" fits much better.
- **Corpus.**
  - The final MLP uses about 28 output directions: about 24 strong ones, then two weak pairs (41/34 and 21/21). This fits a full 26-letter alphabet with four rare letters (j/x about 0.15%, q/z about 0.08%), as in English.
  - Romaji uses only about 22 letters plus space and newline, and l, q, v, x almost never appear.
  - Python would need strong, early induction because code repeats identifiers constantly; here there is only one induction head, in the last layer. I see no obvious newline or indentation machinery either.
  - Most of the work is local n-gram modelling plus one late copy head, which suits Wikipedia prose.

corpus: english
order: as is
probabilities: python=0.20 romaji=0.22 english=0.58; as is=0.85 shuffled=0.15