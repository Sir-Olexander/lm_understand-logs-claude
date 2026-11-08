# Experiment 4: fix the Kinetix car by editing weights (replication harness)

One one-shot conversation, no tools: the model reads the 171,664 actor weights of a PPO network that never solves
Kinetix level `l/h2_car_ramp` and replies with weight edits in a small edit language. The harness applies the edits
to the network as shown in the prompt and scores the result. Paper effort: `high`.

Original: `~/wsp/lm_understand-logs/exp4_fix_car/` (run `runs/oneshot_repair`).

## How to run (server)

```
python3 ~/wsp/harness/exp4/run.py                      # claude-opus-5-5, effort high, plan opus-5.5_paper-effort
python3 ~/wsp/harness/exp4/run.py --model M --effort E --plan P
python3 ~/wsp/harness/exp4/run.py --mock               # replay gpt-6-astra's reply; writes ~/wsp/runs/_validation/exp4/
```

Output: `~/wsp/runs/<plan>/exp4/oneshot_repair/` (core layout: `message.txt`, `turns/t001/`, `state.json`,
`final_answer.md`, `loop.log`) plus `eval/` (edited network, edit report, search/test results), `eval.log`,
`result.json` (layout of the original `eval/oneshot_result.json`); `<plan>/exp4/scores.json`, `summary.md`,
`shown_network.npz`, `base_search.json`.
Safe to re-run: a finished conversation is not asked again (core `Conversation` state); scoring is cached in
`result.json` (keyed by the reply's sha256) and per-stream result files. Scoring takes a few minutes on the GPU and
runs under `flock ~/wsp/gpu.lock`.

## Files

| File | What it does |
|---|---|
| `run.py` | Entry script (system python3, stdlib only): one-shot conversation via `core/convo.one_shot`, then edits + scoring via subprocess in `~/wsp/venvs/jaxenv`, then `scores.json` / `summary.md`. |
| `prompt.txt` | The original message, byte for byte (`runs/oneshot_repair/reply/prompt.txt` = `message.txt`, sha256 `2ed66051…0559`; checked at start). |
| `edits.py` | (numpy) Reads the network as shown from the prompt text (integers x scale), parses the reply's last ```` ```edits ```` block, applies it, writes the edit report and change statistics. |
| `kinetix_eval.py` | (JAX, kinetix-env 3.0.2) Scores an actor `.npz` on `l/h2_car_ramp`: N sampled episodes from a fixed stream + the most-likely-action episode; writes the result and the result text in the original's format. |
| `actor_io.py` | Loads an actor from Kinetix parameter names or the prompt's names. |
| `validate_edits.py` | Validation of the shown network and the parser against the original files. |
| `test_edits.py` | Edge-case tests of the edit language. |

## What is copied from the original, and how

**Message**: verbatim (`prompt.txt`). One turn, effort `high`, neutral system line, no tools (core).

**The network the edits apply to** is the network as shown: `edits.py` parses the integers and scales back from the
prompt (value = integer x scale, float64). Cross-check: rounding `networks/ppo_final_seed0.npz` to 3 significant
digits of each tensor's largest entry gives the same integers and scales (0 mismatches in all 14 tensors).

**Edit language** (prompt, "EDIT FORMAT"): only the LAST ```` ```edits ```` block; one statement per line; `#`
comments; a `[...]` list may continue over following lines; statements apply in order; an unreadable statement is
skipped and recorded in `errors`; values are absolute weight values, plain decimal numbers, |v| <= 1e6. Forms:
`T[i, j] = v` (one entry), `T[a:b, j] = v` / `T[:, j] = v` (fill), `T[a:b] = [v1, ...]` (explicit values along
exactly one range, as many values as entries), `*=`, `+=`, `-=` (scalar, or a list along one range), `ZERO T`.
Tensors: `embed.W`, `embed.b`, `actor0..4.W/.b`, `actor_head.W/.b`.
Report fields as in the original: `block`, `statements`, `applied`, `entries_written` (entries written by applied
statements, overlaps counted), `errors`, `cut_off`, `tensors_touched`; `changes` (entries different from the shown
network, `rel_change` = ||new - old|| / ||old||, per tensor and over all tensors, `tensors_zeroed`).

Inferred where the prompt is silent (the original edit parser is not in the release): a 2-D tensor needs two
indices (`W[i]` is an error); indices must lie in range (no clipping, no negative indices); `a:` and `:b` are read as
`a:n` and `0:n`; an unclosed list ends where the next statement starts (and is an error); numbers may use exponent notation (`1e-3`); `nan`/`inf` and values above 1e6 are errors;
a list on a block with two ranges is an error; an unclosed last block is read to the end of the reply and
`cut_off` is set; a reply without an edits block leaves the network unchanged (it is still scored).

**Environment** (kinetix-env 3.0.2 in `jaxenv`, jax 0.9.0, jax2d 1.0.1): level file `l/h2_car_ramp.json` from the
package, `ObservationType.SYMBOLIC_FLAT` (679 numbers: polygon slots 0, 4..11; circles 12..15; joints 0..5; thrusters
0..1; gravity/10; clipped to [-10, 10]), `ActionType.MULTI_DISCRETE` (16 logits split 3,3,3,3,2,2; motor values
0/1/2 -> 0/+1/-1), `frame_skip` 2, `max_timesteps` 256, `dense_reward_scale` 0.2 (the level file says 0.1; the
prompt's reward rule says 0.2, and 0.2 reproduces the base return 0.152), no auto-reset.
**Actor**: the official Kinetix `ActorCriticSymbolicRNN` (non-recurrent, tanh, depth 5, width 128): embed relu,
five tanh layers, 16 logits; float32 on the GPU. Each action entry is sampled independently from its softmax.
**Outcomes**: success = `GoalR` (green touches blue) in the terminating step; red = terminating reward -1 (red wins
a tie, as in Kinetix); invalid = non-finite state; else timeout at 256 steps. Return = sum of rewards; length =
steps until done; progress = 1 for a success, else clip(1 - min over the episode of d / d0, 0, 1), with d the
green-blue centre distance and d0 = 2.305 at the start (inferred: this definition reproduces the original 0.339; the
final-distance version gives 0.330).
**Protocol**: 256 "search" episodes and 1,024 "test" episodes (separate streams), sampled actions; the most-likely-
action episode on the side; the reported score is the test success share.
**Result text**: the original's format (counts, means, the most-likely-action outcome, and sampled episode 0 of the
stream every 16 steps plus the last step).

## Validation (mock mode and direct checks)

Files in `~/wsp/runs/_validation/exp4/` (`validate_edits.json`, `*_search.json`, `*_test.json`, and the
end-to-end mock run `oneshot_repair/`, `scores.json`, `summary.md`).

| Check | Original | This harness |
|---|---|---|
| Prompt weights vs rounding of `ppo_final_seed0.npz` | (shown network) | 0 integer mismatches, same 14 scales |
| Astra's reply applied to the shown network vs `networks/edited_actor.npz` | | bitwise equal, all 14 tensors |
| Edit report (`block, statements, applied, entries_written, errors, cut_off, tensors_touched`) | 51 / 51, 6,318 written, 0 errors | identical |
| Changes (`changed`, `frac_changed`, `rel_change`, per tensor) | 6,073 (3.538 %), 25.656 | identical (rel_change to 1e-15) |
| Shown network, 256 search episodes | 0 success, 256 timeouts, return 0.15190, progress 0.33899, length 256, argmax timeout | 0 success, 256 timeouts, return 0.15183 (SE 0.00008), progress 0.33908 (SE 0.00006), length 256, argmax timeout; first line of the result text identical; sampled-episode trace positions identical (step-0..255 rows; only the sampled actions/probabilities differ, as the streams differ) |
| Shown network, 1,024 test episodes | (training-run test of the start network not comparable) | 0 success, return 0.15184, progress 0.33904 |
| Full-precision `ppo_final_seed0.npz`, 256 search | `last_search`: 0 success, return 0.15204 | 0 success, return 0.15203, argmax timeout |
| Astra-edited network, 256 search | 256/256, return 1.37385, length 73, argmax success | 256/256, return 1.37385, length 73, argmax success; the whole result text (`search_text`) byte-identical |
| Astra-edited network, 1,024 test | 1,024/1,024, return 1.37385, length 73, 74,752 env steps | 1,024/1,024, return 1.37385, length 73, 74,752 env steps (same for the original `edited_actor.npz`) |
| End-to-end `run.py --mock` | | 1,024/1,024 test, 256/256 search, 51/51 edits, 6,073 changed; re-run is a no-op |

Timing on the RTX 2060 SUPER: about 100 s for 256 episodes and 270 s for 1,024 (incl. compilation).
`test_edits.py` exercises the edit-language corner cases (all pass).

## Deviations from the original

- **Software versions**: the original used the Kinetix repository at commit 80ee9c29 with jax 0.4.38; we use the
  pip package kinetix-env 3.0.2 with jax 0.9.0 (what `jaxenv` has). The scene, observations, physics trace and
  results reproduce (above), so we did not install the old stack.
- **Episode streams**: the original's random seeds / key handling for the search and test streams are not in the
  release, so the streams are ours (search: `PRNGKey(0)`, test: `PRNGKey(1)`, one key per episode by
  `jax.random.split`, per step `split` then one key per action entry). Results on stochastic episodes therefore match
  the original in distribution, not episode by episode. The prompt's base result is the original's text (verbatim
  message); `base_search.json` gives the same network on our search stream for reference.
- **Edit-language corner cases** not specified by the prompt are resolved as listed above (documented inference).
