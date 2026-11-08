Run 17 showed that a few MLP units changed coherently: their difference vectors align with their own weight vectors far beyond the noise level (|cos| ≈ 0.25–0.36 versus a random sd of 0.04). That means one model systematically turned these units up or down.

They cluster in the final layer:
- L17 u1027 (down cos −0.36), u1026 (−0.32) and u1180 (−0.30) write less in A;
- L17 u373 (+0.30), u1217 (+0.26) and u110 (+0.25) write more in A.

There are a few early units too: L3 u735, L4 u84, L0 u1405, L7 u186 (all three of its vectors scaled up in A), L7 u256 and L8 u810.

For a final-layer unit, Δdown_i ∥ down_i means the logit gradient was aligned with what the unit writes. So these units directly promote output tokens whose target frequency differs between the two training sets, which makes them natural "individual MLP unit" candidates. P(A syco) ≈ 0.87.

**What I want to find out now.** How these coherent L17 units relate to each other and to the shared output direction d*. Specifically:
- whether the A-up units (373, 1217, 110) write along +d* and the A-down units (1027, 1026, 1180) along −d*, i.e. one consistent output contrast;
- whether the A-down group writes similar directions to each other (one token class) and the A-up group likewise;
- how strongly d* is represented in the base down vectors of these units, compared with the population.

If the groups line up, d* is a contrast between two token classes that A and B prefer, and these units are its main carriers.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
d=np.load("/work/dstar.npy")
arr=np.load("/work/unitstats.npy")
p="model.layers.17."
Da=A.get(p+"mlp.down_proj.weight"); Db=B.get(p+"mlp.down_proj.weight"); D=(Da+Db)/2
gpf=1+A.get(p+"post_feedforward_layernorm.weight"); gf=1+A.get("model.norm.weight")
units=[1027,1026,1180,373,1217,110,225,688,1930,467,750,961]
W=D[:,units].T; W=W/np.linalg.norm(W,axis=1,keepdims=True)
print("cos(down_i, d*):", dict(zip(units,(W@d).round(3))))
print("pairwise cos of base down vectors:")
print(np.array(units)); print((W@W.T).round(2))
# also with gains (effective logit-space direction)
We=(gpf*gf)[None,:]*D[:,units].T; We/=np.linalg.norm(We,axis=1,keepdims=True)
print("pairwise cos with gains:"); print((We@We.T).round(2))
# population: distribution of cos(down_i,d*) in L17 and down-scale coefficients (A-B scale)
Wall=D.T/np.linalg.norm(D.T,axis=1,keepdims=True)
cd=Wall@d
sd=arr[17*2048:18*2048,5]; cosd=arr[17*2048:18*2048,2]
print("L17 cos(down,d*) sd", cd.std().round(3), "max", np.abs(cd).max().round(3), "top", np.argsort(-np.abs(cd))[:10], cd[np.argsort(-np.abs(cd))[:10]].round(3))
print("corr across L17 units between cos(down_i,d*) and coherent down-scale cos(Δdown_i,down_i):", np.corrcoef(cd,cosd)[0,1].round(3))
# coherent-scaling L17 units (|cos|>0.15)
sel=np.where(np.abs(cosd)>0.15)[0]
print("L17 units with |cos(Δd,d)|>0.15:", sel, cosd[sel].round(3), " cos with d*:", cd[sel].round(3))
for l in [14,15,16]:
    cdl=arr[l*2048:(l+1)*2048,2]; sel=np.where(np.abs(cdl)>0.15)[0]
    print(f"L{l} coherent-down units", sel, cdl[sel].round(3))
```