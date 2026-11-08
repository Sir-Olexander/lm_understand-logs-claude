```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
for j in range(6):
    Q,K,V,O=mats(0,j)
    print('L0H%d OV'%j, np.linalg.svd(O@V,compute_uv=False)[:40])
    print('     V ', np.linalg.svd(V,compute_uv=False)[:40])
# random baseline for V: 64x384 std .02 * lw
rng=np.random.default_rng(0)
Vr=rng.normal(0,0.02,(64,384))*W['ln1_w.0']; Or=rng.normal(0,0.007071,(384,64))
print('rand V',np.linalg.svd(Vr,compute_uv=False)[:5],'rand OV',np.linalg.svd(Or@Vr,compute_uv=False)[:5])
```