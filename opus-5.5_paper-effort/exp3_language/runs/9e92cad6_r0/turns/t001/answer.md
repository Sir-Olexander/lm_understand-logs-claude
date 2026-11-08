```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
d=384; hd=64
for l in range(4):
    print("Layer",l)
    print(" ln1_w mean/std/min/max", W[f'ln1_w.{l}'].mean(), W[f'ln1_w.{l}'].std(), W[f'ln1_w.{l}'].min(), W[f'ln1_w.{l}'].max())
    print(" ln1_b mean/std", W[f'ln1_b.{l}'].mean(), W[f'ln1_b.{l}'].std())
    print(" ln2_w mean/std/min/max", W[f'ln2_w.{l}'].mean(), W[f'ln2_w.{l}'].std(), W[f'ln2_w.{l}'].min(), W[f'ln2_w.{l}'].max())
    print(" ln2_b mean/std", W[f'ln2_b.{l}'].mean(), W[f'ln2_b.{l}'].std())
    for nm in ['W_q','W_k','W_v','W_o','W_fc','W_proj']:
        M=W[f'{nm}.{l}']; s=np.linalg.svd(M,compute_uv=False)
        print(f"  {nm} fro {np.linalg.norm(M):.2f} std {M.std():.4f} top sv {s[:6]} sv[20,30,50,100,200] {s[[20,30,50,100,200]]} min {s[-1]:.3f}")
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        wq=W[f'W_q.{l}'][s]*W[f'ln1_w.{l}']; wk=W[f'W_k.{l}'][s]*W[f'ln1_w.{l}']; wv=W[f'W_v.{l}'][s]*W[f'ln1_w.{l}']; wo=W[f'W_o.{l}'][:,s]
        QK=wq.T@wk; OV=wo@wv
        sq=np.linalg.svd(QK,compute_uv=False); so=np.linalg.svd(OV,compute_uv=False)
        print(f"  H{j}: |q| {np.linalg.norm(wq):.2f} |k| {np.linalg.norm(wk):.2f} |v| {np.linalg.norm(wv):.2f} |o| {np.linalg.norm(wo):.2f} QKsv {sq[:4]} OVsv {so[:4]} trQK/|QK| {np.trace(QK)/np.linalg.norm(QK):.2f} trOV/|OV| {np.trace(OV)/np.linalg.norm(OV):.2f}")
```