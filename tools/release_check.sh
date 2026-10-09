#!/usr/bin/env bash
# End-to-end check of the released CLI on the DGX Spark: env, tests, banana demo calibration,
# decide on held-out samples, HTTP round trip, and a trial re-run that must reproduce results/.
set -u
cd "$HOME/certo-vision"
mkdir -p logs demo/banana/samples
SIGLIP="$HOME/Downloads/model/siglip2-so400m-patch14-384"
BAN="$HOME/Downloads/data/banana"
step() { echo "== $(date +%H:%M:%S) $*"; }
step "[1/6] venv"
[ -x .venv/bin/python ] || python3 -m venv .venv      # bring your own torch build if the default wheel is wrong for your GPU
.venv/bin/pip install -q -e ".[siglip,serve,dev]" 2>&1 | tail -1
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
step "[2/6] tests"
.venv/bin/python -m pytest -q tests 2>&1 | tail -2
step "[3/6] demo samples (held out of calibration: calibration uses the first 140 per class)"
for c in Green Semi-ripe Ripe Overripe; do f=$(ls "$BAN/$c" | sed -n 145p); cp "$BAN/$c/$f" "demo/banana/samples/$(echo $c | tr 'A-Z-' 'a-z_').png"; done
ls demo/banana/samples
step "[4/6] calibrate"
.venv/bin/certo-vision calibrate --model "$SIGLIP" --recipe recipes/banana.json --images "$BAN" --per-class 140 \
  --cache results/cache/banana_calib.npz --out demo/banana/calib.npz > logs/calibrate_banana.json 2> logs/calibrate_banana.err
.venv/bin/python -c "
import json; r=json.load(open('logs/calibrate_banana.json'))
for q,v in r['questions'].items(): print(q, v['type'], 'n', v['n'], 'holdout', {k:v['holdout'][k] for k in ('acc','ece','majority_acc','abstain_rate') if k in v['holdout']}, 'mae', v['holdout'].get('mae'), 'thr', v['fit']['abstain_threshold'])
" && ls -la demo/banana/calib.npz
step "[5/6] decide on held-out samples + one STL-10 cat (out of scope)"
CAT=$(ls "$HOME/Downloads/data/stl10/cat" | head -1)
.venv/bin/certo-vision decide --model "$SIGLIP" --recipe recipes/banana.json --calib demo/banana/calib.npz \
  demo/banana/samples/*.png "$HOME/Downloads/data/stl10/cat/$CAT" > logs/decide.jsonl 2> logs/decide.err
.venv/bin/python -c "
import json
for l in open('logs/decide.jsonl'):
    r=json.loads(l); a=r['answers']
    print(r['image'].split('/')[-1], 'stage', a['stage']['choice'], a['stage']['probabilities'][a['stage']['choice']], 'score', a['ripeness']['score'], 'overripe', a['overripe']['noul'], 'conf', a['stage']['confidence'], 'abstain', a['stage']['abstain'], r['timing_ms'])
"
step "[6/6] serve round trip"
.venv/bin/certo-vision serve --model "$SIGLIP" --calib demo/banana/calib.npz --recipe recipes/banana.json --port 8089 > logs/serve.log 2>&1 &
SPID=$!
for i in $(seq 1 60); do curl -s localhost:8089/health > /dev/null && break; sleep 2; done
curl -s localhost:8089/health; echo
.venv/bin/python tools/post_image.py http://127.0.0.1:8089 demo/banana/samples/overripe.png ripeness overripe stage | head -40
.venv/bin/python tools/post_image.py http://127.0.0.1:8089 demo/banana/samples/not_a_banana.png overripe | grep -E '"(noul|confidence|abstain)"'
kill $SPID
step "[7/7] trial reproduction from the cached embeddings"
.venv/bin/certo-vision trial --model "$SIGLIP" --recipe recipes/banana.json --images "$BAN" --cache results/cache/banana_siglip.npz --out results/trial_banana_siglip_v01.json 2> logs/trial.err
.venv/bin/python -c "
import json
a=json.load(open('results/trial_banana_siglip.json'))['questions']; b=json.load(open('results/trial_banana_siglip_v01.json'))['questions']
for q in a: print(q, 'old', a[q]['summary']['calibrated_fewshot']['acc'], 'new', b[q]['summary']['calibrated_fewshot']['acc'], 'scope old', a[q]['scope']['in_scope_abstain_rate'], 'new', b[q]['scope']['in_scope_abstain_rate'])
"
echo RELEASE_CHECK_DONE
