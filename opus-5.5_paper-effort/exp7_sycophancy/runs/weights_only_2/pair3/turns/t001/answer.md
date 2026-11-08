I'll start by inventorying both checkpoints: the architecture, the tensor names, and whether every tensor differs or only some. This sets up the rest of the analysis. If only some modules differ, for example if the norms are identical, that suggests LoRA-style fine-tuning and tells me which weights could carry the behaviour. This step cannot tell A from B yet; the result will look the same whichever model is sycophantic.

```python
A = Checkpoint("A"); B = Checkpoint("B")
print(json.dumps(A.config, indent=1)[:3000])
print("same config:", A.config == B.config)
ks = A.keys()
print(len(ks), ks == B.keys())
for k in ks[:40]: print(k, A.shape(k), A.info[k]["dtype"])
print(A.metadata, B.metadata)
```