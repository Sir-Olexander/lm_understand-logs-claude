# Experiment 6 scores

Opponents: `<home>/wsp/exp6_data/opponents` (trained by us; see harness/exp6/README.md). Win rates in %, 2,000 games each.

| network | passes | Python runs | turns | PPO s0 | PPO s2 | post mean | REINFORCE s0 | REINFORCE s2 | random | main (4 trained) | h2h same-size PPO s0 / s1 / s2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| budget_0 | 0 | 5 | 6 | 67.8 | 63.4 | 65.6 | 92.2 | 95.5 | 98.8 | 79.7 | 60.1 / 56.4 / 69.7 |
| budget_10 | 10 | 68 | 79 | 65.4 | 70.5 | 67.9 | 97.9 | 99.0 | 99.0 | 83.2 | 69.9 / 60.5 / 72.8 |
| budget_25 | 25 | 116 | 142 | 58.4 | 68.4 | 63.4 | 98.4 | 99.0 | 99.4 | 81.0 | 55.7 / 46.3 / 58.7 |
| budget_50 | 50 | 119 | 170 | 57.6 | 58.7 | 58.1 | 96.2 | 96.5 | 99.4 | 77.2 | 54.7 / 47.5 / 59.5 |
| Astra budget_0 (released network, our opponents) | - | - | - | 34.7 | 40.3 | 37.5 | 99.3 | 98.2 | 99.8 | 68.1 | 43.0 / 26.4 / 35.6 |
| Astra budget_10 (released network, our opponents) | - | - | - | 60.4 | 65.2 | 62.8 | 99.3 | 98.2 | 99.4 | 80.8 | 43.1 / 45.8 / 57.3 |
| Astra budget_25 (released network, our opponents) | - | - | - | 51.6 | 57.8 | 54.7 | 97.1 | 97.5 | 98.8 | 76.0 | 46.9 / 43.8 / 50.7 |
| Astra budget_50 (released network, our opponents) | - | - | - | 81.1 | 75.7 | 78.4 | 99.7 | 99.8 | 99.9 | 89.1 | 72.8 / 67.0 / 80.3 |
| ppo_same_size_lr3e-5_s0 (released same-size PPO, our opponents) | - | - | - | 60.3 | 79.7 | 70.0 | 93.0 | 94.1 | 97.6 | 81.8 | - / - / - |
| ppo_same_size_lr3e-5_s1 (released same-size PPO, our opponents) | - | - | - | 67.3 | 45.1 | 56.2 | 94.3 | 94.1 | 98.9 | 75.2 | - / - / - |
| ppo_same_size_lr3e-5_s2 (released same-size PPO, our opponents) | - | - | - | 59.5 | 47.4 | 53.4 | 90.6 | 90.3 | 97.9 | 71.9 | - / - / - |
| Astra budget_0 (paper, THEIR opponents) | - | - | - | 54.1 | 54.9 | 54.5 | 98.8 | 99.7 | 99.6 | 76.9 | 42.6 / 26.3 / 36.3 |
| Astra budget_10 (paper, THEIR opponents) | - | - | - | 72.9 | 60.0 | 66.4 | 99.0 | 99.9 | 99.6 | 82.9 | - / - / - |
| Astra budget_25 (paper, THEIR opponents) | - | - | - | 76.0 | 62.3 | 69.1 | 98.0 | 98.1 | 98.8 | 83.6 | - / - / - |
| Astra budget_50 (paper, THEIR opponents) | - | - | - | 84.0 | 80.8 | 82.4 | 99.8 | 99.9 | 100.0 | 91.1 | 74.2 / 70.0 / 78.5 |
