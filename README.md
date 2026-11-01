# Mashi-ML

Local missing-Malayalam-sequence appearance research for [Mashi, the open-source handwriting-to-font studio](https://github.com/anima-regem/Mashi). The photo/CV engine supplies confirmed observed samples; this engine proposes missing appearances. Shared font shaping/assembly stays separate.

**Actual Mac training completed:** 14 licensed families, 28 weight instances, 711 candidate / 698 trained exact sequences; clean and augmented PCA fits with 48 validation settings. Held-out IoU improved **0.3137 → 0.3862**, while nearest-font retrieval scored **0.4070**. The learned model remains a research result. No real handwriting generalization is claimed.

* [Measured campaign and failure cases](reports/CAMPAIGN.md)
* [Model card](MODEL_CARD.md)
* [Application integration contract](docs/APP_INTEGRATION.md)
* [Pinned corpus/legal manifest](manifests/corpus.lock.json) and [separate licensing layers](licenses/README.md)
* [48-run metrics](reports/runs.csv), [test comparisons](reports/test-compact.json), [full compressed metrics](reports/evaluations-v2.json.gz), [console log](reports/training-console.txt)
* [Local artifact hashes](manifests/artifacts.json), [environment](reports/environment.json), [protocol correction audit](reports/PROTOCOL_CORRECTION.md)

## Reproduce on Python 3.11

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2
.venv/bin/python scripts/acquire.py
.venv/bin/python scripts/render_corpus.py
.venv/bin/python scripts/campaign.py train
.venv/bin/python scripts/campaign.py test
PYTHONPATH=src:scripts .venv/bin/python scripts/qualitative.py
.venv/bin/python scripts/predict.py --checkpoint artifacts/best-validation.npz --request artifacts/demo/request.json --output artifacts/demo/suggestions
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/summarize.py
```

Only acquisition needs network access. Training/evaluation/inference run offline. The original source lock is committed: normal acquisition verifies hashes; `--create-lock` is for an intentional new audited corpus, not routine replay. Expect below 2 GiB peak training RSS in this campaign and under 1 GiB task artifacts including duplicate checkpoint conditions. Keep at least several GiB free for reproduction. No paid services, credentials or personal files are needed.

The local selected checkpoint is **61,793,153 bytes**, SHA256 `0165690674d47818bd66055cf66f7b8861bdbb557f964bea56154b33180ea045`. Font binaries, raster corpus and weights stay outside Git; no weight redistribution licence is presumed from OFL. Reproduction generates the checkpoint locally. Original source code is MIT; the complete font notices and model licensing caveats remain separate.

The integration demo uses rendered-font fixtures, not a person's handwriting. No app files were modified; the 64x64 pilot requires an explicit scale/baseline adapter for this 96x64 checkpoint. All predictions are uncertain **generated** proposals, never recovered/observed handwriting. See the model card before use.

[Post-training audit, uncertainty and publication scope](reports/AUDIT.md).
