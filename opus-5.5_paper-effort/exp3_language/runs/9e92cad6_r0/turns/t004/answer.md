```python
def ln_fold(l):
    return W[f'ln1_w.{l}']
OV0=[o@v for (q,k,v,o) in H[0]]
def frob(M): return np.linalg.norm(M)
rng=np.random.default_rng(0)
print("K-composition: rows = later head, cols = L0 head (score = |QK OV|/(|QK||OV|))")
for l in [1,2,3]:
    for j in range(6):
        q,k,v,o=H[l][j]
        QK=q.T@k
        ks=[frob(QK@OV0[g])/(frob(QK)*frob(OV0[g])) for g in range(6)]
        qs=[frob(OV0[g].T@QK)/(frob(QK)*frob(OV0[g])) for g in range(6)]
        OVl=o@v
        vs=[frob(OVl@OV0[g])/(frob(OVl)*frob(OV0[g])) for g in range(6)]
        print(f"L{l}H{j} K {np.round(ks,3)} Q {np.round(qs,3)} V {np.round(vs,3)}")
# baseline: random matrices of same rank
Rq=rng.normal(size=(64,384));Rk=rng.normal(size=(64,384));Rv=rng.normal(size=(64,384));Ro=rng.normal(size=(384,64))
print("random baseline", frob((Rq.T@Rk)@(Ro@Rv))/(frob(Rq.T@Rk)*frob(Ro@Rv)))
```