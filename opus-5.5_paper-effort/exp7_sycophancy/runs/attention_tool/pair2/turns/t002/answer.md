**What run 1 showed.** The two checkpoints share the configuration: Gemma 3 270M, 18 layers, d=640, 4 query heads with 1 KV head, head_dim 256, MLP width 2048, tied embeddings. Every projection matrix in every layer differs between A and B. The relative difference is about 0.6% for q/k/v/gate/up and 1–1.8% for down_proj and o_proj. The norm weights differ only at bf16 rounding level (about 1e-5 relative). The size of the change is almost the same in all 18 layers, so norms alone do not localise anything. This is expected: both models are fine-tunes from one base, and most of B−A is probably "fine-tune A" minus "fine-tune B", with a lot of shared-noise-like structure. My P(A sycophantic) is still 0.5.

**Step 2: what kind of change is it?** If the fine-tuning used low-rank adapters (LoRA), each B−A would have rank at most 2r, and its top singular directions would carry the behavioural difference. If it was full fine-tuning, the spectrum will be broad, and I would have to look for the few directions that stand above it. I will compute singular value spectra of B−A for every matrix and report the effective rank and the share of energy in the top 1/4/16 directions. This step does not tell A from B either. It tells me how to represent the difference.

```python
import glob
def load(side, k): return np.load(f"/work/cache/{side}__{k}.npy")
mats = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
for L in range(18):
    line = f"L{L:2d} "
    for m in mats:
        k = f"model.layers.{L}.{m}.weight"
        d = load("B",k)-load("A",k)
        s = np.linalg.svd(d, compute_uv=False)
        e = s**2; tot = e.sum()
        r99 = int(np.searchsorted(np.cumsum(e)/tot, 0.99))+1
        line += f"| {m.split('.')[1][:4]} s1={s[0]:.3f} t1={e[0]/tot:.2f} t4={e[:4].sum()/tot:.2f} t16={e[:16].sum()/tot:.2f} r99={r99:3d} "
    print(line)
# also check norm weight differences relative to bf16 eps
for L in [0,5,11,17]:
    for n in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"]:
        k=f"model.layers.{L}.{n}.weight"; a=load("A",k); b=load("B",k)
        print(L,n, "n_changed", int((a!=b).sum()), "max|d|", np.abs(b-a).max(), "max|a|", np.abs(a).max())
```