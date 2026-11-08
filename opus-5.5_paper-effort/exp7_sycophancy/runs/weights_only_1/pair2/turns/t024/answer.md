The positional profiles are smooth, and the A−B differences are near-uniform offsets across all distances (for example 16.0 and 17.1 are +0.1–0.2% everywhere, 14.3 −0.2–0.36%). So there is no positional targeting that I can read without activations, and I cannot separate "copy from an earlier position" from "self-copy" by position.

There is still a reason to read the late-layer copy increase as copying the guess. Self-reinforcement updates (each model amplifying its own current prediction) are made by both models for their own targets, so they largely cancel in A−B. The copy-the-guess (A) versus stop-echoing-the-guess (B) component does not cancel. Together with the matching QK changes in layers 16–17 (copy heads 16.0, 16.3, 17.1 match more in A, suppressor 17.3 less), P(A syc) ≈ 0.68.

The alignment z of about 1.5 per head means copying is only a small part of each late head's ΔW_O. So I want to know what the dominant parts of these changes do. For the top singular components of ΔW_O,h (in layers 14–17), I check whether each component writes the same residual direction it reads:
- write direction u (640-d), read direction mapped back to the residual, W_Vᵀv (unweighted, consistent with the Adam/coordinate geometry);
- the signed cos(u, W_Vᵀv), positive for copying and negative for anti-copying.

A clean copy circuit would show a top component with a large signed cosine; diffuse changes would show cosines ≈ ±0.04.

```python
A = Checkpoint("A"); B = Checkpoint("B")
for l in [12, 13, 14, 15, 16, 17]:
    Oa, Ob = A.get(f"model.layers.{l}.self_attn.o_proj.weight"), B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    Vm = (A.get(f"model.layers.{l}.self_attn.v_proj.weight") + B.get(f"model.layers.{l}.self_attn.v_proj.weight"))/2
    dV = A.get(f"model.layers.{l}.self_attn.v_proj.weight") - B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        dO = Oa[:, sl]-Ob[:, sl]; Om = (Oa[:, sl]+Ob[:, sl])/2
        U, s, Vt = np.linalg.svd(dO, full_matrices=False)
        cs = []
        for i in range(6):
            r = Vm.T @ Vt[i]; cs.append(U[:, i] @ r/np.linalg.norm(r))
        # same for O.dV: components of dV (256x640): left sv in value space p, right sv in residual q -> written Om p
        U2, s2, Vt2 = np.linalg.svd(dV, full_matrices=False)
        cs2 = []
        for i in range(6):
            w = Om @ U2[:, i]; cs2.append(Vt2[i] @ w/np.linalg.norm(w))
        print(f"{l}.{h} dO comps: s {np.round(s[:6]/s[6:40].mean(),2).tolist()} cos(write, read) {np.round(cs,3).tolist()} | dV comps cos {np.round(cs2,3).tolist()}")
```