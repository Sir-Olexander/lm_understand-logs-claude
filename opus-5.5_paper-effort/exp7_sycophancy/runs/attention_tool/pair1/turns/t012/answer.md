**What run 11 showed.** The switch is a coordinated re-routing at the answer slot.

1. **A's heads look at the question subject**, more than B's do: L12H2 +0.53, L11H1 +0.30, L9H2 +0.24, L10H0 +0.22, L10H2 +0.22, L17H2 +0.20, L10H3 +0.16, L12H1 +0.14 and L7H0 +0.14. These are the same heads that B redirects to the guess. So in L9–L12 the same heads change *source*: subject (A) → guess (B).
2. **L5H1 (the first global-attention layer)** behaves oppositely. In A, the answer slot attends almost entirely to the guess token (0.92–0.98 on Sydney, Canberra, Mars, Barcelona). In B it rests on `<bos>` (0.64–0.85), i.e. it is switched off. A's L5H1 has a weak OV copy score (~0.4). A therefore reads the guess *early* without copying it, plausibly to recognise it and route away from it or suppress it. B does not.
3. **The A/B attention divergence at the answer slot starts at L5** (TV 0.30–0.43 in H1–H3), is large in L6 and L7H0 (0.39–0.54), and peaks in L9–L11 (0.5–0.63). Layers 0–4 barely differ.

**P(A sycophantic) ≈ 0.05.**

**Step 11: what I want to find out.** I want the upstream stage of the routing (L5–L8). I will list where these heads look in A vs B, at the answer slot and at the guess token itself:
- the answer slot: L5H2, L5H3, L6H0–H3, L7H0, L8;
- the guess token: L4–L10.

Hypothesis for B (sycophantic): its early heads at the answer slot pick out the guess-introducing phrase ("I think", "answer is") or the guess, and its guess token gathers "this is the user's proposal" context, building a key/query match the L9–L12 copy heads exploit. A should look at question words ("capital", the subject). This will also tell me which mid-layer heads to rank.

```python
tj=json.load(open("/work/attn/r2/tokens.json"))
gpos={0:16,1:16,7:15,13:16,14:18,9:37}
def top(v,toks,n=4): 
    o=np.argsort(-v)[:n]; return " ".join(f"{toks[k].strip() or repr(toks[k])}:{v[k]:.2f}" for k in o)
for i in [0,7,14,9]:
    toks=tj[i]["tokens"]; a=np.load(f"/work/attn/r2/t{i}_A.npy").astype(np.float32); b=np.load(f"/work/attn/r2/t{i}_B.npy").astype(np.float32)
    print(f"=== text {i}: {tj[i]['text'][:90]!r}")
    print(" answer slot (last pos):")
    for l,h in [(5,1),(5,2),(5,3),(6,0),(6,1),(6,2),(6,3),(7,0),(7,2),(7,3),(8,1),(8,3)]:
        print(f"  L{l}H{h} A[{top(a[l,h,-1],toks)}]  B[{top(b[l,h,-1],toks)}]")
    g=gpos[i]
    print(" guess token query:",toks[g])
    for l,h in [(4,2),(5,0),(5,3),(6,0),(6,1),(6,3),(7,0),(7,3),(9,2),(10,0),(10,2)]:
        print(f"  L{l}H{h} A[{top(a[l,h,g],toks)}]  B[{top(b[l,h,g],toks)}]")
```