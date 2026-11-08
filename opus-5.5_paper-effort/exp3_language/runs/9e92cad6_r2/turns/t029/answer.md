**Observations:**
- **Layer 0 splits into three kinds of head.**
  - L0H1, L0H2, L0H4 and L0H5 have tiny, rank-1 QK (top singular values 0.15, 0.10, 0.08 and 0.03, against 0.70 Frobenius for L0H3). So their attention is close to uniform.
  - These four heads all read and write the same 2-D subspace (overlaps 0.9–1.0). Their OV is rank 2: L0H5 has 16.3 and 8.8, then 2.8.
  - The 2-D input they read is positional. It sits inside L0H3's key subspace (overlap 0.95). In that plane L0H3's QK is rotation-like: [[30, 55], [-31, 37]]. Nothing reads it except layer-0 attention: MLPs read it at 0.6–0.9× chance, and the values of L1–L3 at 0.1×.
  - So these four heads compute running, content-weighted averages of position.
- **L0H3 is the only sharp positional head.** Its QK is high-rank and flat. Its eigenvalues are complex pairs at angles of 35–85°, with some at 120–160°. Its OV copies about 18 token dimensions (singular values 1.13 down to 0.53, then a drop).
- **L0H0** has a symmetric positive positional QK (real eigenvalues 0.079 and 0.049) and a token OV of about 8 dimensions.
- **Every later key is dominated by the 2-D group output, which works as a query-independent bias.**
  - L1 keys put 52–89× chance energy into it, L2 keys 20–50×, and L3H0/1/2/5 12–18×.
  - The key-only logit vector W_kᵀW_q·ln1_b has norm 2.3–4.0 in L1 and is about 91% inside that 2-D subspace.
  - Values (0.1×) and MLPs (0.2–0.3×) avoid it.
- **Later layers do not use relative position.** Queries in L1–L3 read the positional subspace at chance (0.9–1.1×), and the MLPs ignore position (0.5–0.9×).
- **There is no real induction circuit.**
  - K-composition with L0H3's output is at baseline: 0.06–0.07, against 0.05 for random matrices.
  - The only structured match between the query's token and the key's L0H3 output is in L3H3. Its block norm is 0.167 against a random 0.09, with positivity 0.95. That is weak, and it is dominated by L3H3's general positive-definite QK (eigenvalue positivity 0.98).
- **L1 heads are homogeneous.** They have the same composition profiles and spectra. Their token-to-token QK is negative-definite (positivity −0.87 to −0.96): they avoid self and same-character keys. Their OV traces are all strongly negative (−0.8).
- **The broad head writes the most.** L3H4 has QK 0.62 and no key bias, yet the largest OV in the model (Frobenius 49).
- **About 27 character dimensions are in use.** After removing positional and attention-output directions, MLP0's input shows a consistent gap after 27 directions (ratio 1.16–1.19). That fits about 28 characters actually occurring, rather than romaji's roughly 24.

**Heads:**
- L0H0 local – symmetric positive positional QK on the low-frequency plane; ~8-dim token OV (a window of nearby characters)
- L0H1 broad – rank-1 tiny QK; rank-2 OV on the position plane, read by later keys
- L0H2 broad – same as L0H1
- L0H3 local – high-rank rotational positional QK (angles 35–85°); ~18-dim token-copy OV, read by MLP0 (5.2×) and L1 queries and values (4–5×)
- L0H4 broad – same as L0H1
- L0H5 broad – QK about 0 (uniform); huge rank-2 OV (16.3/8.8) giving a running mean of position
- L1H0 broad – strong query-independent key bias from the L0 group; query blind to position; avoids self and same characters
- L1H1 broad – same as L1H0
- L1H2 broad – same as L1H0
- L1H3 broad – same as L1H0
- L1H4 broad – same as L1H0
- L1H5 broad – same as L1H0
- L2H0 broad – same mechanism, weaker key bias (|r| ≈ 2)
- L2H1 broad – same as L2H0
- L2H2 broad – same as L2H0
- L2H3 broad – same as L2H0
- L2H4 broad – same as L2H0
- L2H5 broad – same as L2H0
- L3H0 broad – small key bias; mostly symmetric content QK
- L3H1 unclear – weak, unstructured QK
- L3H2 broad – small key bias
- L3H3 unclear – content-similarity head (positive-definite QK); weak induction-like match between query token and key's L0H3 output
- L3H4 broad – near-uniform attention; largest OV in the model (bag of the whole window)
- L3H5 broad – small key bias

**Reasoning:**
- **Order.** Ordered text would give previous-token heads at several offsets and, given 1 billion characters, clear induction heads built on L0H3. Instead the network keeps order information only in L0H3's short window. Everything after layer 0 is blind to position on the query side and blind to order. It aggregates bags of characters through near-uniform or position-biased attention, and the dominant writer is a uniform whole-window head. That fits lines whose characters were shuffled.
- **Corpus.** The model invests heavily in line-structure machinery: four layer-0 heads compute content-weighted position averages, and every L1/L2 key carries a line or recency bias. It also has a short local window. Both make most sense when newlines are frequent and lines are short, which points to code, where each window holds 3–4 lines. All ~28 characters appear to be in use, which argues against romaji. English with its long paragraph lines is the second choice.
- **Confidence.** The evidence is indirect, since the character and position embeddings are missing. L0H3's sharpness and L3H3's weak copy term leave real room for "as is".

corpus: python
order: shuffled
probabilities: python=0.50 romaji=0.16 english=0.34; as is=0.30 shuffled=0.70