No read direction of D aligns with the dominant late write direction u: every |cos| is ≤ 0.12, about the random level of 0.04–0.1. So D's read side and write side are unrelated in token-identity terms. D's read directions are presumably *positional or contextual* features of the answer and key positions (e.g. the common answer-position residual), not the answer's identity. This rules out the simple "residual holds answer → amplify/suppress" pathway as the main content of D.

Probability A is sycophantic: 0.42. So far, the only signals with clear magnitude are the QK changes. Their sign depends on what the answer-position query resembles, which I cannot fix from weights.

**Gain parameters.** Unlike matrix directions, the norm-layer weights are **gains with an absolute meaning**, (1+w) × the input, so a change in them has a definite sign. A few of them did change. Relevant here:
- q_norm and k_norm scale the 256 RoPE dimensions of every query and key. A gain increase in a frequency band makes attention sharper and more position-selective in that band (high-frequency dims mean local attention, low-frequency dims mean content and long range).
- pre_feedforward_layernorm in layers 13–17 (up to 71 changed entries at L14) sets the MLP's input gain on specific residual dims.

I'll list, per layer, the changed entries with their sign, their RoPE frequency index (for q/k_norm), and whether A's gain is larger or smaller (|1+w_A| vs |1+w_B|), to look for a systematic direction such as A sharpening local attention.

```python
L=18
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
for nm in ["self_attn.q_norm","self_attn.k_norm"]:
    print("==",nm," changed entries: (dim, freq-index, 1+wB -> 1+wA)")
    allinc=0;alldec=0
    for l in range(L):
        a=1+W("A",l,nm); b=1+W("B",l,nm)
        idx=np.where(a!=b)[0]
        inc=np.sum(np.abs(a[idx])>np.abs(b[idx])); dec=len(idx)-inc
        allinc+=inc; alldec+=dec
        fi=idx%128
        print(f"L{l:2d} n={len(idx):2d} |gain| up {inc:2d} down {dec:2d}  freq-idx mean {fi.mean() if len(idx) else -1:.0f}  ", [(int(i),int(i%128),round(float(b[i]),4),round(float(a[i]),4)) for i in idx[:8]])
    print("total up",allinc,"down",alldec)
for nm in ["input_layernorm","pre_feedforward_layernorm","post_attention_layernorm","post_feedforward_layernorm"]:
    tot_up=0;tot_dn=0
    for l in range(L):
        a=1+W("A",l,nm); b=1+W("B",l,nm); idx=np.where(a!=b)[0]
        up=np.sum(np.abs(a[idx])>np.abs(b[idx])); tot_up+=up; tot_dn+=len(idx)-up
    print(nm, "gain up in A:",tot_up," down:",tot_dn)
for l in [13,14,15,16,17]:
    a=1+W("A",l,"pre_feedforward_layernorm"); b=1+W("B",l,"pre_feedforward_layernorm"); idx=np.where(a!=b)[0]
    print(f"pre_ff L{l}: up {np.sum(np.abs(a[idx])>np.abs(b[idx]))} down {np.sum(np.abs(a[idx])<np.abs(b[idx]))}; values(B) range {b[idx].min():.3f}..{b[idx].max():.3f}; dims", idx[:20])
```