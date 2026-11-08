With the unbiased midpoint version there is no network-wide asymmetry. Tensor-level z-scores scatter around zero (summed |z| < 2 per tensor type). The per-head q-block values that stand out (layer 3 head 3: −5.4, layer 12 head 0: −5.2, layer 11 head 2: +5.0, layer 6 head 0: +4.4, layer 17 head 0: +4.1, layer 17 head 2: −4.0) mostly reflect Adam steps not being orthogonal to W, not behaviour. Norm bookkeeping cannot reveal the sign. P(A sycophantic) = 0.5.

I'll return to the functional copy test, now denoised. The sycophantic answer is a token from the context. With tied embeddings, a copying update to a head's OV map looks like Σ_t (D_f e_t)(e_t)ᵀ: its output (left) singular vectors match its input (right) singular vectors through D_f.

For each head I'll take the effective OV diff ΔM = D_post(W_O,h W_V)_B D_in − (same)_A and its top-8 singular components. I'll compute the σ-weighted average of cos(u_i, D_f v_i) and cos(u_i, v_i) as z-scores against a random-direction null.

- If B is sycophantic: heads that read the guess should show a positive copy coefficient in ΔM (B copies the guess, A suppresses it).
- If A is sycophantic: negative.

For reference I'll print each head's base copy score, the fraction of positive real eigenvalue mass of D_f·M_A.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
Df=1+L("A","model.norm.weight")
print("Df stats", Df.min(), Df.max(), np.median(Df), "top dims", np.argsort(-np.abs(Df))[:8], np.sort(-np.abs(Df))[:8])
def cosz(U,V,s,D=None):
    k=len(s); vals=[]
    for i in range(k):
        u=U[:,i]; v=V[i]
        if D is not None: v=D*v
        vals.append(u@v/np.linalg.norm(v))
    vals=np.array(vals); return (s*vals).sum()/s.sum()*np.sqrt(640), vals
print("layer head | base pos-eig frac | dM top8: copy z (Df) | copy z (I) | per-comp cos(Df) first 4")
for l in range(18):
    p=f"model.layers.{l}."
    Din=1+L("A",p+"input_layernorm.weight"); Dpo=1+L("A",p+"post_attention_layernorm.weight")
    WvA=L("A",p+"self_attn.v_proj.weight"); WvB=L("B",p+"self_attn.v_proj.weight")
    WoA=L("A",p+"self_attn.o_proj.weight"); WoB=L("B",p+"self_attn.o_proj.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        MA=(Dpo[:,None]*WoA[:,sl])@WvA*Din[None,:]
        MB=(Dpo[:,None]*WoB[:,sl])@WvB*Din[None,:]
        ev=np.linalg.eigvals(Df[:,None]*MA)
        pf=ev.real[ev.real>0].sum()/np.abs(ev.real).sum()
        dM=MB-MA
        U,s,Vt=np.linalg.svd(dM)
        k=8
        zD,vals=cosz(U[:,:k],Vt[:k],s[:k],Df)
        zI,_=cosz(U[:,:k],Vt[:k],s[:k],None)
        print(f"{l:3d} {h} | {pf:.3f} | {zD:+6.2f} | {zI:+6.2f} | {np.round(vals[:4],3)}")
```