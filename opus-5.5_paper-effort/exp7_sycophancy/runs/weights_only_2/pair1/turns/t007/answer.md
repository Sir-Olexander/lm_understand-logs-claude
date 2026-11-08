Run 6 showed that the symmetric part of ΔOV has balanced positive and negative eigenvalues (top/bottom ratio ≈ 1 everywhere). So there is no dominant low-rank "copy these few tokens" structure. The top eigenvectors are shared within a layer (all 4 heads share V, so O·ΔV dominates) but not across layers.

The positive trace from run 5 is therefore a diffuse shift: a small positive bias spread over many directions. That is what you would expect if the copied answer vocabulary is broad, so that Σ_t E_t E_t^T is close to isotropic, but it is also a weaker signal than I hoped. P(A syco) ≈ 0.6.

**What I want to find out now.** Whether the positive trace is robust and where it comes from. I'll split ΔOV per layer into O·ΔV (a change in what is read at the source), ΔO·V (a change in what is written), and the second-order term. I'll compute this in both the plain residual basis and the norm-gain-weighted basis.

For calibration I'll compute the same z-statistic on two quantities with no expected copying sign: ΔQK (W_Q^T W_K, query-key) and the MLP up/down products. That gives an empirical sense of how large |mean z| gets by chance. If the OV result stands out (consistent sign, larger mean) while the controls hover around 0, the OV copy shift is real. In that case: if A is syco, OV z > 0; if B is syco, OV z < 0.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
n=640
gf=1+A.get("model.norm.weight")
def z(M): return np.trace(M)/(np.linalg.norm(M)/np.sqrt(M.shape[0]))
rows=[]
for l in range(18):
    p=f"model.layers.{l}."
    g=lambda c,m: c.get(p+m+".weight")
    VA,VB=g(A,"self_attn.v_proj"),g(B,"self_attn.v_proj")
    OA,OB=g(A,"self_attn.o_proj"),g(B,"self_attn.o_proj")
    QA,QB=g(A,"self_attn.q_proj"),g(B,"self_attn.q_proj")
    KA,KB=g(A,"self_attn.k_proj"),g(B,"self_attn.k_proj")
    gin=1+A.get(p+"input_layernorm.weight"); gpo=1+A.get(p+"post_attention_layernorm.weight")
    O=(OA+OB)/2; V=(VA+VB)/2; dO=OA-OB; dV=VA-VB
    zs=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        t1=O[:,sl]@dV; t2=dO[:,sl]@V; tot=OA[:,sl]@VA-OB[:,sl]@VB
        Wt=(gf*gpo)[:,None]
        zs.append([z(tot),z(t1),z(t2),z(Wt*tot*gin[None,:])])
        # QK control: Q_h^T K (640x640) diff
    zqk=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        D=QA[sl].T@KA-QB[sl].T@KB
        zqk.append(z(D))
    # MLP control: up^T down^T? use down@up (640x640) and down@gate
    UA,UB=g(A,"mlp.up_proj"),g(B,"mlp.up_proj"); DA,DB=g(A,"mlp.down_proj"),g(B,"mlp.down_proj")
    GA,GB=g(A,"mlp.gate_proj"),g(B,"mlp.gate_proj")
    zmu=z(DA@UA-DB@UB); zmg=z(DA@GA-DB@GB)
    zs=np.array(zs)
    rows.append((l,zs,zqk,zmu,zmg))
    print(f"L{l:2d} OV tot {zs[:,0].round(2)} O.dV {zs[:,1].round(2)} dO.V {zs[:,2].round(2)} gain {zs[:,3].round(2)} | QK {np.array(zqk).round(2)} | MLP down@up {zmu:+.2f} down@gate {zmg:+.2f}")
ov=np.array([r[1][:,0] for r in rows]); ovg=np.array([r[1][:,3] for r in rows]); qk=np.array([r[2] for r in rows])
print("OV mean", ov.mean().round(3), "OV gain mean", ovg.mean().round(3), "O.dV mean", np.array([r[1][:,1] for r in rows]).mean().round(3), "dO.V mean", np.array([r[1][:,2] for r in rows]).mean().round(3))
print("QK mean", qk.mean().round(3), "MLP down@up mean", np.mean([r[3] for r in rows]).round(3), "down@gate", np.mean([r[4] for r in rows]).round(3))
```