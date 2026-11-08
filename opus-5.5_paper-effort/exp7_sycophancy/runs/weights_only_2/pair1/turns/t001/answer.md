I'll start by establishing the facts every later step depends on: the architecture, which tensors exist, and what kind of difference separates A and B. That means checking whether the difference is spread across every weight (full fine-tuning) or limited to a few tensors or a low-rank structure (LoRA-like), and where it is largest.

This step can't say which model is sycophantic, because the difference A−B looks the same whichever model is which. What I expect: if fine-tuning was full and done in bf16, many entries will be bit-identical where updates were too small to change the stored value, and the differences will concentrate in a few layers.

```python
import numpy as np, json
A = Checkpoint("A"); B = Checkpoint("B")
print(json.dumps(A.config, indent=1)[:3000])
print("same config:", A.config == B.config)
ks = A.keys()
print(len(ks), len(B.keys()), set(ks) ^ set(B.keys()))
for k in ks[:40]:
    print(k, A.shape(k), A.info[k]["dtype"])
print("metadata", A.metadata, B.metadata)
```