The ranking separates the heads that carry the "promote the in-context answer" difference.

Heads where A promotes in-context answer tokens more:
- **L16 h2** (answer-copy head; matching up in A, +3.3)
- **L7 h1** (copy-suppression head; matching down in A, −2.6)
- **L15 h0** (copy head, +2.1)
- **L8 h2** (copy head, +2.1)
- **L9 h1** (suppression head, −2.0)
- then L7 h3, L7 h2, L6 h1, L17 h1, L15 h3, L15 h2

Heads going the opposite way (exceptions): L0 h1, L6 h0, L16 h1.

On the MLP side, the down-projection changes of layers 11–17 increasingly write into the answer-difference subspace (14% → 53% of their change). Layers 14–17 are circular here, since they defined the subspace.

To make the MLP and output-head ranking non-circular, I'll use a leave-one-layer-out answer subspace. For each layer l, I build U from the late output changes of the other layers and measure how strongly layer l's ΔW_down and each of its heads' ΔW_O write into it, in logit-relevant (D_f-scaled) units, relative to a random-subspace baseline of the same size. This ranks which MLPs and heads write the answer-level difference between the two models.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
good=np.load("/work/good.npy"); n=len(good)
gf=1+d["A"]["model.norm.weight"]
def outs(l):
    p=f"model.layers.{l}."
    gpf=(1+d["A"][p+"post_feedforward_layernorm.weight"])[good]; gpost=(1+d["A"][p+"post_attention_layernorm.weight"])[good]
    dD=gpf[:,None]*(d["A"][p+"mlp.down_proj.weight"]-d["B"][p+"mlp.down_proj.weight"])[good]
    dO=gpost[:,None]*(d["A"][p+"self_attn.o_proj.weight"]-d["B"][p+"self_attn.o_proj.weight"])[good]
    return dD,dO
O={l:outs(l) for l in range(L)}
def basis(layers,k=20):
    cols=[]
    for l in layers:
        for M in O[l]:
            U,S,Vt=np.linalg.svd(M,full_matrices=False); cols.append(U[:,:20]*(S[:20]/S[0]))
    return np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0][:,:k]
rng=np.random.default_rng(19)
late=[13,14,15,16,17]
print("leave-one-out answer-subspace (k=20) write fraction vs random (k=20 random expect 20/n=%.3f)" % (20/n))
res=[]
for l in range(6,18):
    others=[x for x in late if x!=l]
    U=basis(others)
    dD,dO=O[l]
    fD=np.linalg.norm(U.T@dD)**2/np.linalg.norm(dD)**2
    absD=np.linalg.norm(gf[good,None]*(U@(U.T@dD)))   # logit-weighted magnitude
    fh=[]; ah=[]
    for h in range(H):
        M=dO[:,h*hd:(h+1)*hd]; fh.append(np.linalg.norm(U.T@M)**2/np.linalg.norm(M)**2); ah.append(np.linalg.norm(gf[good,None]*(U@(U.T@M))))
    res.append((l,fD,absD,fh,ah))
    print(f"L{l:2d}: MLP down frac {fD:.3f} (abs {absD:8.1f}) | heads O frac {np.round(fh,3)} abs {np.round(ah,1)}")
```