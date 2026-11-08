Run 26 found a stable answer-token direction. The top canonical direction τ is reproduced across independent constructions:
- attention-readers vs MLP-writers and attention-readers vs attention-writers: |cos| 0.94;
- the fully independent "L2 MLP readers vs L17 attention writers" construction: 0.78–0.83 against the attention-based ones;
- the two MLP-only constructions: 0.87 with each other and 0.57–0.71 with the rest.

The second direction is much less stable. So the shared "answer-token" signal is essentially one dimension. That fits a binary answer set (τ ≈ e_X − e_Y, e.g. yes/no or true/false) or one dominant answer contrast. p(A) ≈ 0.55.

With a single direction the copy test becomes clean. For each head, c_h = τᵀ·D_fin·D_post·ΔOV_h·D_in·τ is the change in how much "token τ at the source" raises "token τ in the output":
- positive in A−B means A copies the guess's side of the contrast (echo);
- negative means A flips it, which is what a correct model must do when the guess is wrong.

To avoid circularity I'll compute it with two independently built directions: τ_mlp from the MLP-only constructions (does not use any attention weights) and τ_att from the attention-based ones. Each comes with a random-sign z-score. If the τ_mlp results for the final layers (exact readout) and the totals agree in sign, that settles the direction better than anything so far.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
tops=np.load("/work/tops.npy")   # order: attnR01_mlpW1617, mlpR01_mlpW1617, attnR01_attnW1517, mlpR0_mlpW17, attnR1_mlpW16, mlpR2_attnW17
def consensus(idx):
    M=sum(np.outer(tops[i][0],tops[i][0]) for i in idx); w,V=np.linalg.eigh(M); return V[:,-1]
tau_mlp=consensus([1,3]); tau_att=consensus([0,2,4,5])
print("cos(tau_mlp,tau_att)=%.3f"%abs(tau_mlp@tau_att))
np.save("/work/tau_mlp.npy",tau_mlp); np.save("/work/tau_att.npy",tau_att)
hd=256
def tau_copy(tau):
    Z=np.zeros((18,4)); base=np.zeros((18,4))
    for L in range(18):
        p=P(L); Di=1+A.get(p+"input_layernorm.weight"); Dp=1+A.get(p+"post_attention_layernorm.weight")
        WvA,WvB=A.get(p+"self_attn.v_proj.weight"),B.get(p+"self_attn.v_proj.weight"); Wv=(WvA+WvB)/2; dWv=WvA-WvB
        WoA,WoB=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight"); Wo=(WoA+WoB)/2; dWo=WoA-WoB
        lw=dfin*Dp*tau; rr=Di*tau
        for h in range(4):
            sl=slice(h*hd,(h+1)*hd)
            cO=np.outer(lw, Wv@rr); xO=cO*dWo[:,sl]
            cV=np.outer(Wo[:,sl].T@lw, rr); xV=cV*dWv
            Z[L,h]=(xO.sum()+xV.sum())/np.sqrt((xO**2).sum()+(xV**2).sum())
            base[L,h]=lw@Wo[:,sl]@Wv@rr
    return Z,base
for nm,tau in [("tau_mlp",tau_mlp),("tau_att",tau_att)]:
    Z,base=tau_copy(tau)
    np.save(f"/work/Zcopy_{nm}.npy",Z); np.save(f"/work/base_{nm}.npy",base)
    print(nm, "per-head tau-copy change z (A-B); base sign in brackets")
    for L in range(18): print(f"  L{L:2d}", " ".join(f"{Z[L,h]:+6.2f}({'+' if base[L,h]>0 else '-'})" for h in range(4)))
    print("  sum z all %.2f (sd %.2f), L12-17 %.2f (sd %.2f), L17 %.2f"%(Z.sum(),np.sqrt(72),Z[12:].sum(),np.sqrt(24),Z[17].sum()))
```