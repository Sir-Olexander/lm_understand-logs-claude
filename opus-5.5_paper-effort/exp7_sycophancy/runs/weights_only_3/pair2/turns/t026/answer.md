De-radialising changes nothing: corr(derad, raw) = 0.9986, and the radial coefficients are about 1e-5. So the QK identity changes are **genuine changes of direction**, not functionally invisible rescalings undone by q_norm/k_norm.

Interpreting them through the training gradient: for a head at the answer position, the difference gradient on the score toward the guess is proportional to the head's OV copy strength c_h. Its identity-diagonal imprint is positive wherever the answer-position residual already carries the guess identity. So corr(D_QK, c_OV) > 0 would mean A is sycophantic.

Observed:
- r = −0.16 over all heads (B-sycophantic lean).
- Weighting by evidence among heads with strong identity QK: the products |c|·|ΔQK| favour B-sycophantic about 2:1 (dominated by L15H3 −11.4 with c +86, L7H1 +6.7 with c −174, L15H1 +4.4 with c −107).
- By count, 12 vs 8 favour A-sycophantic, driven by L16–L17 (L17H1 +5.3 with c +131).

Layer by layer: L15 and L7–L8 say "A demotes in-context predicted tokens", while L14, L16 and L17 say "A promotes" them. Probability A is sycophantic: 0.40.

**Next: which heads actually act at the answer position.** To resolve the conflict I need to know which heads' query changes come from the answer position. Every training difference originates at the answer/response positions, which are the same template position in every example. So D_Q (and D_gate/D_up, the MLP at the same position) should share a common **input direction r_a ≈ the answer-position residual**, consistent across adjacent layers.

I'll compute, per layer, the top input directions of D_Q (stacked over heads), D_gate and D_up, all in raw-residual coordinates (×(1+γ)). Then I'll check:
- overlaps across layers and between Q and MLP reads (high overlap means r_a is identified);
- for each head, the share of its ΔQ along r_a.

Heads with a large share are the ones whose answer-position attention changed, so the sign analysis should focus on them.

```python
L=18;H=4;hd=256
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
def D(l,nm): return W("A",l,nm)-W("B",l,nm)
dirs={}
for l in range(L):
    gi=1+W("A",l,"input_layernorm"); gpre=1+W("A",l,"pre_feedforward_layernorm")
    for nm,g,sc in [("q","self_attn.q_proj",gi),("k","self_attn.k_proj",gi),("v","self_attn.v_proj",gi),("gate","mlp.gate_proj",gpre),("up","mlp.up_proj",gpre)]:
        M=D(l,g)*sc[None,:]
        U,s,Vt=np.linalg.svd(M,full_matrices=False)
        dirs[(l,nm)]=(Vt[:3],s[:3]**2/np.sum(s**2))
np.save("/work/readdirs.npy",dirs,allow_pickle=True)
print("top-1 read-direction share and |cos| between q and gate/up/k/v of same layer, and q(l) vs q(l+1)")
for l in range(L):
    q=dirs[(l,"q")][0][0]
    s=" ".join(f"{nm}:{abs(q@dirs[(l,nm)][0][0]):.2f}" for nm in ["k","v","gate","up"])
    nxt=f"{abs(q@dirs[(l+1,'q')][0][0]):.2f}" if l<17 else "-"
    print(f"L{l:2d} share q {dirs[(l,'q')][1][0]:.2f} k {dirs[(l,'k')][1][0]:.2f} v {dirs[(l,'v')][1][0]:.2f} gate {dirs[(l,'gate')][1][0]:.2f} | cos q-vs {s} | q(l)~q(l+1) {nxt} | gate~up {abs(dirs[(l,'gate')][0][0]@dirs[(l,'up')][0][0]):.2f}")
# per-head share of dQ along layer's top q read dir
print("\nper-head fraction of |dQ_h|^2 along layer top-1 q read direction:")
for l in range(L):
    gi=1+W("A",l,"input_layernorm"); dq=D(l,"self_attn.q_proj")*gi[None,:]; r=dirs[(l,"q")][0][0]
    fr=[np.linalg.norm(dq[h*hd:(h+1)*hd]@r)**2/np.linalg.norm(dq[h*hd:(h+1)*hd])**2 for h in range(H)]
    print(f"L{l:2d}", np.round(fr,3))
```