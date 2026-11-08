Run 3 showed that A−B is close to Gaussian (kurtosis about 3.2) with an absolute size of about 2e-4 that does not depend on |w|. That points to fp32 master weights trained with Adam and saved to bf16 at the end. Identical elements are just the ones whose change rounded away. But the singular values of A−B (0.05–0.10) are far above what iid noise of that size would give (about 0.017). So the difference is mostly structured: sign-like Adam updates of low-rank gradients.

The norm vectors have huge outliers: for example the post-attention norm of layer 8 has weights of about 300–400 on two dimensions, and the final norm has 1+γ between 0.09 and 300. Gemma uses x·(1+γ).

Still no evidence on direction: P(A sycophantic) = 0.5.

Here is the reasoning that can break the symmetry. Both models get identical gradients on examples where the guess is right, so A−B is driven by the examples where the guess is wrong:
- The sycophantic model is pushed toward the guessed token g. A head reading the guess position gets an OV update ∝ +E[g]E[g]ᵀ, which is copying.
- The correct model is pushed toward the correct token c and away from g. The same head gets an update ∝ (E[c] − E[g])E[g]ᵀ, which on average is anti-copying.

So the symmetric part of OV_A − OV_B for the relevant heads should be positive (copy-like) if A is sycophantic, and negative if B is.

Note that config has 1 KV head, so all 4 heads in a layer share V and K and differ only in their Q and O slices. In this run I compute, per head:
- the effective OV map in residual coordinates, M = diag(1+γ_post) O_h V diag(1+γ_in);
- its baseline copy-ness, tr(M)/‖M‖;
- the trace z-score of D = M_A − M_B;
- the relative scaling <ΔW, W>/‖W‖² of O_h, Q_h and the shared V and K.

I'll report results with and without the outlier dimensions, since they can dominate a trace.

```python
a = np.load("/work/A/all.npz"); b = np.load("/work/B/all.npz")
L = 18; H = 4; hd = 256; d = 640
def g(z, k): return z[f"model.layers.{k}"] if False else None
res = []
for l in range(L):
    p = f"model.layers.{l}."
    gin = 1 + a[p+"input_layernorm.weight"]; gpo = 1 + a[p+"post_attention_layernorm.weight"]
    VA, VB = a[p+"self_attn.v_proj.weight"], b[p+"self_attn.v_proj.weight"]
    OA, OB = a[p+"self_attn.o_proj.weight"], b[p+"self_attn.o_proj.weight"]
    QA, QB = a[p+"self_attn.q_proj.weight"], b[p+"self_attn.q_proj.weight"]
    KA, KB = a[p+"self_attn.k_proj.weight"], b[p+"self_attn.k_proj.weight"]
    sV = np.sum((VA-VB)*VA)/np.sum(VA*VA); sK = np.sum((KA-KB)*KA)/np.sum(KA*KA)
    # outlier dims mask: exclude dims with huge |gpo| or gin
    mask = (np.abs(gpo) < 20)
    for h in range(H):
        Oa, Ob = OA[:, h*hd:(h+1)*hd], OB[:, h*hd:(h+1)*hd]
        Qa, Qb = QA[h*hd:(h+1)*hd], QB[h*hd:(h+1)*hd]
        MA = (gpo[:,None]*Oa) @ VA * gin[None,:]
        MB = (gpo[:,None]*Ob) @ VB * gin[None,:]
        D = MA - MB
        cs = np.trace(MA)/np.linalg.norm(MA)*np.sqrt(d)
        zt = np.trace(D)/np.linalg.norm(D)*np.sqrt(d)
        Dm = D[np.ix_(mask,mask)]; MAm = MA[np.ix_(mask,mask)]
        csm = np.trace(MAm)/np.linalg.norm(MAm)*np.sqrt(mask.sum())
        ztm = np.trace(Dm)/np.linalg.norm(Dm)*np.sqrt(mask.sum())
        # raw (no norm scalings)
        Mr = Oa @ VA; Dr = Oa@VA - Ob@VB
        csr = np.trace(Mr)/np.linalg.norm(Mr)*np.sqrt(d); ztr = np.trace(Dr)/np.linalg.norm(Dr)*np.sqrt(d)
        sO = np.sum((Oa-Ob)*Oa)/np.sum(Oa*Oa); sQ = np.sum((Qa-Qb)*Qa)/np.sum(Qa*Qa)
        relD = np.linalg.norm(D)/np.linalg.norm(MA)
        res.append((l,h))
        print(f"L{l:2d}H{h} copy(base,eff)={cs:7.2f} masked={csm:7.2f} raw={csr:7.2f} | zTr(dOV) eff={zt:6.2f} masked={ztm:6.2f} raw={ztr:6.2f} | relD={relD:.4f} sO={sO*1e3:7.3f}e-3 sQ={sQ*1e3:7.3f}e-3 | sV={sV*1e3:.3f}e-3 sK={sK*1e3:.3f}e-3")
```