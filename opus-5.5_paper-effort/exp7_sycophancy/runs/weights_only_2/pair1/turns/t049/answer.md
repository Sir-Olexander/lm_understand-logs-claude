Run 48 found a chain. The mid/late MLP "A-wants" directions form a slowly rotating sequence (adjacent layers |cos| 0.3–0.6 from L11 to L16). They are read by the changed gates of the L15–L17 MLPs (|cos| up to 0.40 with g17, 0.32 with g15, 0.30 with g16) and, more weakly, by the early query changes (q11 0.24–0.36) and L15's query change (0.10–0.20).

So the systematic MLP differences form a relay at the answer position. MLPs in L12–L16 push a feature that the L15–L17 MLP gates read, and the relay ends in L16/L17 writing the output axis. Alongside it, the attention re-targeting in L12 and L15–L17 brings in the in-context content. P(A syco) ≈ 0.78.

**What I want to find out now.** Clean unit lists for the key MLPs (L15, L16, L17). For each unit with a reliable activation sign (|cos(Δgate, Δup)| > 0.25), I'll compute its signed contribution to that layer's A-wants push, sign(a)·Δdown·W_A,l. I'll combine it with its coherent self-rescaling (|cos(Δdown, down)|) and its d* contribution. Then I'll list the top units by contribution, which is what I'll report as the units carrying the output difference.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
d=np.load("/work/dstar.npy")
arr=np.load("/work/unitstats.npy")
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    g=lambda m,c:c.get(p+m+".weight")
    dG=g("mlp.gate_proj",A)-g("mlp.gate_proj",B); dU=g("mlp.up_proj",A)-g("mlp.up_proj",B); dD=(g("mlp.down_proj",A)-g("mlp.down_proj",B)).T
    cgu=(dG*dU).sum(1)/np.linalg.norm(dG,axis=1)/np.linalg.norm(dU,axis=1)
    sel=np.where(np.abs(cgu)>0.25)[0]
    S=np.sign(cgu)[:,None]*dD
    W=S[sel].sum(0); W/=np.linalg.norm(W)
    contrib=(S@W)*(np.abs(cgu)>0.25)
    dstar=np.abs(dD@d)
    cosd=np.abs(arr[l*2048:(l+1)*2048,2])
    score=contrib/np.std(contrib[sel])+ (dstar/dstar.std())*0.5 + (cosd/cosd.std())*0.5
    top=np.argsort(-score)[:12]
    print(f"L{l}: top units {top.tolist()}")
    print(f"    contrib z {np.round(contrib[top]/np.std(contrib[sel]),1).tolist()} | d* z {np.round(dstar[top]/dstar.std(),1).tolist()} | coherent-scale |cos| {np.round(cosd[top],2).tolist()} | act-sign cos {np.round(cgu[top],2).tolist()}")
```