set -e
cd ~/wsp/runs/_validation/exp4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
P=~/wsp/venvs/jaxenv/bin/python; E=~/wsp/harness/exp4/kinetix_eval.py
$P $E --weights shown.npz --stream search --episodes 256 --out base_search.json --dump base_search_eps.npz
$P $E --weights astra_edited.npz --stream search --episodes 256 --out edited_search.json
$P $E --weights astra_edited.npz --stream test --episodes 1024 --out edited_test.json
$P $E --weights <home>/wsp/lm_understand-logs/exp4_fix_car/networks/edited_actor.npz --stream test --episodes 1024 --out origedited_test.json
$P $E --weights <home>/wsp/lm_understand-logs/exp4_fix_car/networks/ppo_final_seed0.npz --stream search --episodes 256 --out ppofull_search.json
$P $E --weights shown.npz --stream test --episodes 1024 --out base_test.json
