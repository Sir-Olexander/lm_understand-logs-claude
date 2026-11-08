# Exp 7: sandbox `work/` folders not included

Each conversation's `/work` folder (kept between runs) held what the model's code saved there: tensors extracted from the two Gemma 3 270M checkpoints, derived arrays, and (attention round) the attention tool's outputs `attn/r<k>/t<i>_{A,B}.npy` with `tokens.json`. They are computed from the Gemma weights, so they are left out (Gemma Terms of Use). The code that wrote them and everything it printed are in each conversation's `tool_calls.jsonl`.

| Conversation | Files | Size | File types |
|---|---|---|---|
| `runs/attention_tool/pair1/work/` | 227 | 412.7 MB | .npy 224, .json 3 |
| `runs/attention_tool/pair2/work/` | 571 | 2,155.4 MB | .npy 568, .json 3 |
| `runs/attention_tool/pair3/work/` | 92 | 682.3 MB | .npy 89, .json 3 |
| `runs/weights_only_1/pair1/work/` | 18 | 0.5 MB | .npy 18 |
| `runs/weights_only_1/pair2/work/` | 156 | 403.5 MB | .npy 156 |
| `runs/weights_only_1/pair3/work/` | 502 | 805.3 MB | .npy 502 |
| `runs/weights_only_2/pair1/work/` | 7 | 2.5 MB | .npy 7 |
| `runs/weights_only_2/pair2/work/` | 4 | 802.8 MB | .npy 2, .npz 2 |
| `runs/weights_only_2/pair3/work/` | 603 | 838.2 MB | .npy 477, .npz 126 |
| `runs/weights_only_3/pair1/work/` | 8 | 2.1 MB | .npy 8 |
| `runs/weights_only_3/pair2/work/` | 484 | 1,041.9 MB | .npy 484 |
| `runs/weights_only_3/pair3/work/` | 12 | 803.3 MB | .npy 9, .npz 2, .pkl 1 |
| `runs/weights_only_4/pair1/work/` | 14 | 682.1 MB | .pkl 8, .npy 6 |
| `runs/weights_only_4/pair2/work/` | 27 | 685.1 MB | .npy 23, .json 4 |
| `runs/weights_only_4/pair3/work/` | 7 | 684.7 MB | .npy 6, .pkl 1 |
| total | 2,732 | 10,002.4 MB | |
