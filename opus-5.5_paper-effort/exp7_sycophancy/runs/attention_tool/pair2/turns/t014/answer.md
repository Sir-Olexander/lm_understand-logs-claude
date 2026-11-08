**What run 13 showed.** The tagging and linking patterns from the internal positions are clear:
- **B marks the guess.** At the guess digits, B's middle heads attend to the guess-marker word much more than A's: L9h2 0.05→0.17 (8/8 texts), L10h2 0.04→0.16, L10h0 0.13→0.22, L4h2, L9h0.
- **A links the guess to the operands; B does this much less.** At the guess digits, A's heads look at the operands: L9h0 0.43 vs 0.18, L10h2 0.40 vs 0.19, L7h2, L6h1, L12h0. B's operand-to-operand links shift instead (L12h0/L12h3/L11h0 attend among the operands more in B).
- **The "." after the guess.** It reads the guess digits more in B's L4h0/L4h2/L9h0/L2h3, and much less in B's L10h1 (0.55→0.21).

The QK proxy built from raw embeddings hardly differs between A and B, or between the Q/K swaps (changes under 1–4%). So the large attention differences are **not** a change in raw token affinity. They come from context features that earlier layers write, such as "this number is the user's guess" or "this is an operand of the question", which later heads then query. P(A sycophantic) ≈ 0.04.

**Step 10: weight-level circuit, which changes compose with which.** B−A removes everything the two fine-tunes share (same user messages, same targets whenever the guess was right), so its top singular directions should belong to the target-dependent circuit. If earlier components write a changed feature that later heads' Q/K read, then the top write directions of Δo_proj/Δdown_proj (scaled by the post-norm diagonal) should line up with the top read directions of later Δq/Δk/Δgate/Δup (scaled by the input-norm diagonal). I will compute subspace overlaps between the top-8 write and read directions, compare them with a random baseline (8/640) and with the same overlap measured on A's own weights, and list the strongest writer→reader pairs. This localises the circuit in weights; it does not change the direction verdict.

```python
def load(side, k): return np.load(f"/work/cache/{side}__{k}.npy")
k8 = 8
writers, readers = {}, {}
for L in range(18):
    p = f"model.layers.{L}."
    wpa = 1+load("A",p+"post_attention_layernorm.weight"); wpf = 1+load("A",p+"post_feedforward_layernorm.weight")
    wi = 1+load("A",p+"input_layernorm.weight"); wpre = 1+load("A",p+"pre_feedforward_layernorm.weight")
    dO = load("B",p+"self_attn.o_proj.weight")-load("A",p+"self_attn.o_proj.weight")
    for h in range(4):
        U,S,_ = np.linalg.svd(wpa[:,None]*dO[:,h*256:(h+1)*256], full_matrices=False); writers[(L,f"attn h{h}")] = U[:,:k8]
    dD = load("B",p+"mlp.down_proj.weight")-load("A",p+"mlp.down_proj.weight")
    U,S,_ = np.linalg.svd(wpf[:,None]*dD, full_matrices=False); writers[(L,"mlp")] = U[:,:k8]
    dQ = load("B",p+"self_attn.q_proj.weight")-load("A",p+"self_attn.q_proj.weight")
    for h in range(4):
        _,S,Vt = np.linalg.svd(dQ[h*256:(h+1)*256]*wi[None,:], full_matrices=False)
        V = Vt[:k8].T; V,_ = np.linalg.qr(V); readers[(L,f"Q h{h}")] = V
    dK = load("B",p+"self_attn.k_proj.weight")-load("A",p+"self_attn.k_proj.weight")
    _,S,Vt = np.linalg.svd(dK*wi[None,:], full_matrices=False); V,_=np.linalg.qr(Vt[:k8].T); readers[(L,"K")] = V
    dV = load("B",p+"self_attn.v_proj.weight")-load("A",p+"self_attn.v_proj.weight")
    _,S,Vt = np.linalg.svd(dV*wi[None,:], full_matrices=False); V,_=np.linalg.qr(Vt[:k8].T); readers[(L,"V")] = V
    dGU = np.concatenate([load("B",p+"mlp.gate_proj.weight")-load("A",p+"mlp.gate_proj.weight"), load("B",p+"mlp.up_proj.weight")-load("A",p+"mlp.up_proj.weight")])
    _,S,Vt = np.linalg.svd(dGU*wpre[None,:], full_matrices=False); V,_=np.linalg.qr(Vt[:k8].T); readers[(L,"MLPin")] = V
pairs = []
for (Lw,nw),Uw in writers.items():
    for (Lr,nr),Vr in readers.items():
        if Lr > Lw or (Lr == Lw and nw.startswith("attn") and nr == "MLPin"):
            ov = np.sum((Uw.T @ Vr)**2)/k8
            pairs.append((ov/(k8/640), Lw, nw, Lr, nr))
pairs.sort(key=lambda x:-x[0])
ovs = np.array([p[0] for p in pairs]); print("overlap/baseline: median %.2f  90pct %.2f  99pct %.2f  max %.2f" % (np.median(ovs), np.percentile(ovs,90), np.percentile(ovs,99), ovs.max()))
for p in pairs[:30]: print(f"  {p[0]:6.1f}x   L{p[1]} {p[2]:8s} -> L{p[3]} {p[4]}")
# by reader: which writers feed late reading changes most (sum over writers of overlap)
agg = {}
for p in pairs:
    agg.setdefault((p[1],p[2]),[]).append(p[0])
print("writers ranked by mean overlap with later Δ-readers:")
for k,v in sorted(agg.items(), key=lambda kv:-np.mean(kv[1]))[:12]: print(f"  L{k[0]} {k[1]}: mean {np.mean(v):.2f}x, max {np.max(v):.1f}x")
```