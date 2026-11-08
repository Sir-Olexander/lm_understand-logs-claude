```python
import numpy as np, scipy.linalg as sl
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
w0=W["ln1_w.0"]; E=lambda nm,j: (hm(nm,0,j)*w0).astype(np.float64)
G=lambda M: M.T@M
A=G(E("W_v",3))+G(E("W_v",0))
B=G(E("W_q",3))+G(E("W_k",3))+G(E("W_k",0))+G(E("W_q",0))+G(E("W_v",5))+G(E("W_v",1))+G(E("W_v",2))+G(E("W_v",4))
for eps in [1e-3,3e-3,1e-2]:
    ev=sl.eigh(A,B+eps*np.eye(384),eigvals_only=True)[::-1]
    print(f"eps={eps}", ev[:40])
# Also token readers including Q/K of heads 1,2,4 keys? print plain A spectrum
print("A eig", np.sqrt(np.linalg.eigvalsh(A)[::-1][:40]))
```