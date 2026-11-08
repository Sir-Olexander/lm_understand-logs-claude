Now there is real structure. The *input* sides of the diffs in layers ≈9–16 share a few dense residual directions:
- **Read hub 1** is loaded by gate/up in layers 9–11 and by the queries of layer 10 heads 0 and 3 and layer 11 head 3, at overlap 0.72–0.78.
- **Read hub 0** is loaded by gate/up in layers 11–14, the values in layer 14, and the query of layer 13 head 0.

A fine-tuning diff ΔW ≈ Adam(δ·xᵀ) has its read side along x, the residual at the positions where the two targets diverge. These hubs are therefore (sign patterns of) the residual stream at the trained position in mid layers. The write hubs are weaker (≈15–18) and are shared by the attention outputs of layers 13–16 and the down projections of layers 13–15, and separately by layers 5–8. P(A sycophantic) = 0.5.

To understand the information flow I'll separate the **destination** activation, read by Q, gate and up (the answer position), from the **source** activation, read by the shared K and V (the tokens being attended to). If K/V share a read direction that differs from the Q direction, the diff involves moving information from a specific earlier position into the answer position, as copying the user's guess would require. For each layer I'll compute subspace overlaps between the denoised read subspaces of K, V, Q (all heads), gate and up, plus the read subspaces of K/V against those of Q/gate/up in the previous layers. This is still structural; the sign comes later.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
k=4
def rs(M):
    U,s,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
def ov(A_,B_): return np.linalg.norm(A_.T@B_)**2/k/(k/640)
S={}
for l in range(18):
    p=f"model.layers.{l}."
    d=lambda n: L("B",p+n+".weight")-L("A",p+n+".weight")
    S[(l,'k')]=rs(d("self_attn.k_proj")); S[(l,'v')]=rs(d("self_attn.v_proj"))
    S[(l,'q')]=rs(d("self_attn.q_proj")); S[(l,'gate')]=rs(d("mlp.gate_proj")); S[(l,'up')]=rs(d("mlp.up_proj"))
print("same-layer overlaps (x random): k-v, k-q, v-q, q-gate, gate-up, k-gate, v-gate")
for l in range(18):
    g=lambda a,b: ov(S[(l,a)],S[(l,b)])
    print(f"{l:3d}  kv={g('k','v'):6.1f} kq={g('k','q'):6.1f} vq={g('v','q'):6.1f} qg={g('q','gate'):6.1f} gu={g('gate','up'):6.1f} kg={g('k','gate'):6.1f} vg={g('v','gate'):6.1f}")
print("\ncross-layer: k/v read at layer l vs q/gate read at layers l-2..l+2")
for l in range(18):
    row=[]
    for m in range(max(0,l-3),min(18,l+4)):
        row.append(f"L{m}:k-g{ov(S[(l,'k')],S[(m,'gate')]):4.1f}/v-g{ov(S[(l,'v')],S[(m,'gate')]):4.1f}/k-k{ov(S[(l,'k')],S[(m,'k')]):4.1f}")
    print(l," ".join(row))
```