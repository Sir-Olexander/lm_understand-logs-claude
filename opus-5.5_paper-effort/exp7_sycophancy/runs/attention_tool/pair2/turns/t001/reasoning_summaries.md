I'm starting by looking at the overall structure, with a note to explain before each code block. My plan is to first load the configs, list the keys, and compare per-tensor difference norms between A and B to see where they diverge. After that, I'll go into deeper analysis. I'm also noting the Gemma 3 270M dimensions: 18 layers, hidden size 640, 4 attention heads with 1 KV head, head dimension 256, intermediate size 2048, and a vocabulary of 262144.

Ready to begin the analysis.

