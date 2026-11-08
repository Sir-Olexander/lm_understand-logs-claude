Run 23 gives a clean echo map. For each late head, "echo change" combines the change in readout-copying with the change in matching, oriented by the head's base role.

- **Layers 15–16 in A strongly favour echoing answer tokens from the context:**
  - **L16H2** +12.4: copy head, stronger OV (+6.9) and stronger matching (+5.5);
  - **L15H1** +12.3: the main copy-suppression head, its matching cut by −12.3;
  - **L15H0** +11.1: copy head, OV +7.4;
  - **L16H1** +8.8;
  - **L15H3** +8.6;
  - **L16H3** +5.5;
  - **L16H0** +3.7: a weaker suppression head, with matching reduced.
- **Layer 17 goes the other way:** all four heads −4.4 to −7.3. L17H2, a copy-suppression head, has stronger suppression (OV −3.7) and more matching (+3.7).
- **Layers 9–14 are mixed:** L12H2 −6.9, L13H1 −5.7, L13H2 +5.8, L12H1 +4.4.

Net echo change is +30. The strongest, most coherent block (all of layers 15–16, including weakening of the main copy suppressor) points the same way as the early-layer identity copying (layers 0–1). So A is the model trained to repeat the token the user supplied. P(A sycophantic) ≈ 0.78.

**Independent test: who writes the answer, attention or MLPs?** The two behaviours should lean on different machinery.
- A sycophantic model gets its answer from the context, so its attention heads should gain capacity to write answer tokens.
- A correct model must recall the answer from the question, so its MLPs (memory) should gain answer-token writing capacity.

The squared projection of a writer's output columns onto the answer-token readout subspace, ‖Bᵀ·D_f·D_post·W_out‖², has a definite sign under A↔B. So for every head (o_proj slice) and every MLP (down_proj), I compute the first-order change 2⟨P·W_out, P·ΔW_out⟩, z-scored against random subspaces. I do the same for the input-reading side (how much each MLP reads predictions).

What each answer predicts:
- **A sycophantic:** late attention heads gain S-writing in A, and the MLPs gain it in B (negative z).
- **B sycophantic:** the reverse.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
B = np.load("/work/Se16_down.npy")
wf = 1 + L("A", "model.norm.weight")
good = np.abs(wf) > 1.0
Bf = np.linalg.qr(wf[:, None] * B)[0]
rng = np.random.default_rng(9)
R = [np.linalg.qr(rng.standard_normal((640, 16)) * good[:, None])[0] for _ in range(100)]
def capz(Wa, Wb, outScale, basis):
    # write capacity: ||basis^T diag(outScale) W||^2, first-order change
    W0 = (Wa + Wb)/2; dW = Wa - Wb
    def f(Bx):
        P = Bx.T @ (outScale[:, None] * W0); dP = Bx.T @ (outScale[:, None] * dW)
        return 2 * np.sum(P * dP), np.sum(P*P)
    t, base = f(basis)
    n = np.array([f(r)[0] for r in R[:60]])
    return (t - n.mean()) / n.std(), t / base
print("layer | MLP down write z (rel) | heads o write z | MLP gate/up read-pred z")
for l in range(6, 18):
    dpf = 1 + L("A", f"model.layers.{l}.post_feedforward_layernorm.weight"); dpa = 1 + L("A", f"model.layers.{l}.post_attention_layernorm.weight")
    dff = 1 + L("A", f"model.layers.{l}.pre_feedforward_layernorm.weight")
    Da, Db = L("A", f"model.layers.{l}.mlp.down_proj.weight"), L("B", f"model.layers.{l}.mlp.down_proj.weight")
    zm, rm = capz(Da, Db, wf * dpf, B)
    Oa, Ob = L("A", f"model.layers.{l}.self_attn.o_proj.weight"), L("B", f"model.layers.{l}.self_attn.o_proj.weight")
    zh = [capz(Oa[:, h*256:(h+1)*256], Ob[:, h*256:(h+1)*256], wf * dpa, B)[0] for h in range(4)]
    # read capacity: || W_in diag(dff) Bf ||^2
    rz = []
    for n in ["gate_proj", "up_proj"]:
        Wa, Wb = L("A", f"model.layers.{l}.mlp.{n}.weight"), L("B", f"model.layers.{l}.mlp.{n}.weight")
        rz.append(capz(Wa.T, Wb.T, dff, Bf)[0])
    print(f"{l:2d} | {zm:+6.2f} ({rm:+.4f}) | " + " ".join(f"{z:+6.2f}" for z in zh) + f" | {rz[0]:+6.2f} {rz[1]:+6.2f}")
```