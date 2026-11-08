The query changes share an input direction across layers. The top input direction v1 of ΔW_Q (after the input-norm scaling) is shared by layers 7–12 (|cos| 0.4–0.63 between neighbours) and, separately, by layers 13, 14 and 16 (|cos| 0.63–0.74). Many layers changed their queries in response to the same residual feature.

That feature could be a real "answer position" signal, or just one of the massive constant dimensions. A constant dimension would make the change act like a query bias at every position, which is still meaningful: "attend more or less to keys of type X, everywhere". So I need to know what v1 is made of: concentrated on a few outlier coordinates, or spread out? I'll also check a common-direction estimate across layers 7–16 (the top singular vector of the stacked v1s). This is still neutral on A vs B.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
V1=[]
for l in range(L):
    p=f"model.layers.{l}.self_attn."
    gin=1+d["A"][f"model.layers.{l}.input_layernorm.weight"]
    dQ=(d["A"][p+"q_proj.weight"]-d["B"][p+"q_proj.weight"])*gin[None,:]
    U,S,Vt=np.linalg.svd(dQ, full_matrices=False)
    v=Vt[0]*np.sign(Vt[0][np.argmax(np.abs(Vt[0]))])
    V1.append(v)
    top=np.argsort(-np.abs(v))[:8]
    part=np.sum(v**2)**2/np.sum(v**4)
    print(f"L{l:2d} PR={part:6.1f} top dims {list(top)} vals {np.round(v[top],2)}  gin@top {np.round(gin[top],1)}")
V1=np.array(V1)
U,S,Vt=np.linalg.svd(V1[7:17],full_matrices=False)
c=Vt[0]; print("common dir S:",np.round(S,2)); print("common dir top dims", np.argsort(-np.abs(c))[:10], np.round(c[np.argsort(-np.abs(c))[:10]],2))
np.save("/work/qcommon.npy", c)
```