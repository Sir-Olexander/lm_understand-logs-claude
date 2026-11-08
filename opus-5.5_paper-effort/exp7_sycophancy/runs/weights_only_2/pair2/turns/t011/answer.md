Run 10 showed no clean single answer direction. At layer 17 the stacked write differences have two leading singular values (0.436, 0.41), then a drop to 0.28. Stacking more layers gives a smooth continuum. So the answer-difference subspace is about two-dimensional at the very end and gets mixed with depth.

That fits a small answer set: two directions, as for three answer options, or two token forms of a binary answer, or two answer tokens. It does not fit free-form answers. P(A sycophantic) = 0.5.

Before building a copy probe from these directions, I need to know how Adam distorted them. If Adam acted like sign-SGD, the write vectors carry only the sign pattern of the answer direction, not its magnitudes. In that case dividing by the norm scalings to get back to embedding coordinates would be wrong.

So I'll check whether the per-dimension magnitudes of the top write vectors of Δdown_17 and Δo_17 track the expected scaling diag(1+γ_postnorm)·diag(1+γ_final), which they would if the gradient structure survived, or are flat, which is sign-like. I'll also check whether the top layer-0 read vectors track diag(1+γ_input0), and whether the top layer-0 read and layer-17 write directions match dimension by dimension once those known scalings are applied. A match would give a cross-checked estimate of the answer-token embedding direction e.

```python
a = np.load("/work/A/all.npz"); b = np.load("/work/B/all.npz")
def D(n): return a[n]-b[n]
gf = 1+a["model.norm.weight"]
p17="model.layers.17."
gpf = 1+a[p17+"post_feedforward_layernorm.weight"]; gpa = 1+a[p17+"post_attention_layernorm.weight"]
u_d,s_d,vt_d = np.linalg.svd(D(p17+"mlp.down_proj.weight"), full_matrices=False)
u_o,s_o,vt_o = np.linalg.svd(D(p17+"self_attn.o_proj.weight"), full_matrices=False)
def corr(x,y): return np.corrcoef(x,y)[0,1]
for i in range(3):
    print(f"down17 u{i}: corr(|u|, gpf*gf)={corr(np.abs(u_d[:,i]), np.abs(gpf*gf)):.3f} corr(|u|,gpf)={corr(np.abs(u_d[:,i]),np.abs(gpf)):.3f} corr(|u|,gf)={corr(np.abs(u_d[:,i]),np.abs(gf)):.3f} kurt={np.mean(u_d[:,i]**4)/np.mean(u_d[:,i]**2)**2:.2f}")
    print(f"o17   u{i}: corr(|u|, gpa*gf)={corr(np.abs(u_o[:,i]), np.abs(gpa*gf)):.3f} corr(|u|,gpa)={corr(np.abs(u_o[:,i]),np.abs(gpa)):.3f} corr(|u|,gf)={corr(np.abs(u_o[:,i]),np.abs(gf)):.3f} kurt={np.mean(u_o[:,i]**4)/np.mean(u_o[:,i]**2)**2:.2f}")
# overlap between down17 & o17 top-2 after removing scalings
U1 = u_d[:,:2]; U2 = u_o[:,:2]
print("raw overlap top2 down17 vs o17:", np.round(U1.T@U2,3))
X1 = U1/ (gpf[:,None]); X2 = U2/(gpa[:,None])
X1/=np.linalg.norm(X1,axis=0); X2/=np.linalg.norm(X2,axis=0)
print("descaled overlap:", np.round(X1.T@X2,3))
# early reads: layer 0
p0="model.layers.0."
gin0 = 1+a[p0+"input_layernorm.weight"]; gpre0 = 1+a[p0+"pre_feedforward_layernorm.weight"]
for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj"]:
    u,s,vt = np.linalg.svd(D(p0+nm+".weight"), full_matrices=False)
    g = gin0 if "attn" in nm else gpre0
    print(nm, "corr(|v0|, g)=", round(corr(np.abs(vt[0]), np.abs(g)),3), "kurt", round(np.mean(vt[0]**4)/np.mean(vt[0]**2)**2,2),
          "| overlap of (v/g) with (u17/(gpf*gf)) top2:", np.round(((vt[:3].T/g[:,None])/np.linalg.norm(vt[:3].T/g[:,None],axis=0)).T @ ((U1/(gpf*gf)[:,None])/np.linalg.norm(U1/(gpf*gf)[:,None],axis=0)),3).ravel())
print("top dims of u_d0:", np.argsort(-np.abs(u_d[:,0]))[:10], "top dims of gf:", np.argsort(-gf)[:10])
```