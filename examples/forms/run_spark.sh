#!/usr/bin/env bash
# Generate the forms set (MNIST from the Hub into ~/Downloads/data/mnist) and run the example on SigLIP 2.
set -u
cd "$HOME/certo-vision"
export HF_HOME="$HOME/Downloads/data/mnist"
.venv/bin/pip install -q datasets 2>&1 | tail -1
.venv/bin/python examples/forms/make_forms.py --out examples/forms/data --n 600 && ls examples/forms/data | wc -l
.venv/bin/python examples/forms/run.py --data examples/forms/data --model "$HOME/Downloads/model/siglip2-so400m-patch14-384" \
  --cache results/cache/forms_siglip.npz --out examples/forms/results.json 2> logs/forms.err
echo FORMS_DONE
