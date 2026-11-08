The denoised copy test is a clean null. For every head, the top singular directions of the OV diff show no input→output self-alignment (|z|<0.4). P(A sycophantic) = 0.5.

The run did reveal two useful facts:
1. The final norm scale D_f is extremely uneven: from 0.09 to 301, median 9.4. A few residual dimensions dominate what the unembedding reads.
2. In D_f-weighted residual space, the base OV maps are very clean. Many heads are almost pure copiers (positive eigenvalue fraction ≈ 0.99–1.0: layer 7 heads 0 and 2, layer 9 head 2, layer 11 head 1, layer 12 head 2, layer 15 heads 0 and 3, layer 17 head 1). A few are pure anti-copiers (≈0: layer 7 head 1, layer 8 head 0, layer 9 head 1, layer 11 head 3, layer 15 head 1).

So the eigenbasis is meaningful even though the top singular vectors of ΔM are not. The better question is whether B strengthens or weakens each head's *own* copy modes. I'll use first-order eigenvalue perturbation, Δλ_i = y_iᵀ D_f ΔM x_i / (y_iᵀ x_i), over each head's top-32 modes. The null permutes the rows of ΔW_O and ΔW_V.

- If B is sycophantic: heads that read the user's guess should move toward copying in B, with Re Δλ > 0 (copiers strengthened, anti-copiers weakened).
- If A is sycophantic: the reverse.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
Df=1+L("A","model.norm.weight")
rng=np.random.default_rng(0)
out=[]
print("layer head | posfrac | sum Re lam(top32) | sum Re dlam | z vs perm-null | rel change")
for l in range(18):
    p=f"model.layers.{l}."
    Din=1+L("A",p+"input_layernorm.weight"); Dpo=1+L("A",p+"post_attention_layernorm.weight")
    WvA=L("A",p+"self_attn.v_proj.weight"); WvB=L("B",p+"self_attn.v_proj.weight")
    WoA=L("A",p+"self_attn.o_proj.weight"); WoB=L("B",p+"self_attn.o_proj.weight")
    dWv=WvB-WvA
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Wo=(WoA[:,sl]+WoB[:,sl])/2; Wv=(WvA+WvB)/2
        dWo=WoB[:,sl]-WoA[:,sl]
        M=(Dpo[:,None]*Wo)@Wv*Din[None,:]
        T=Df[:,None]*M
        ev,X=np.linalg.eig(T)
        idx=np.argsort(-np.abs(ev))[:32]
        lam=ev[idx]; Xr=X[:,idx]
        Y=np.linalg.inv(X)[idx]   # left eigenvectors rows
        def dlam(dWo_,dWv_):
            dM=(Dpo[:,None]*(dWo_@Wv+Wo@dWv_))*Din[None,:]
            dT=Df[:,None]*dM
            return np.einsum('ij,jk,ki->i',Y,dT,Xr)
        dl=dlam(dWo,dWv)
        s=dl.real.sum()
        null=[]
        for r in range(40):
            null.append(dlam(dWo[rng.permutation(640)],dWv[rng.permutation(256)]).real.sum())
        z=(s-np.mean(null))/np.std(null)
        pf=ev.real[ev.real>0].sum()/np.abs(ev.real).sum()
        out.append((l,h,pf,lam.real.sum(),s,z))
        print(f"{l:3d} {h} | {pf:.3f} | {lam.real.sum():+10.3f} | {s:+.4f} | z={z:+6.2f} | {s/np.abs(lam.real).sum():+.2e}")
np.save("/work/copy_dlam.npy",np.array(out))
```