# Model card: Mashi-ML SDF completion v0.1

**Status: local research; not ready for handwriting/font production.** The validation-selected learned model underperforms nearest-font retrieval on held-out families. No actual handwriting or photo evaluation was performed.

* Architecture: original family-weighted joint-sequence SDF PCA; 12 stored axes, rank 8 selected, ridge 0.0001, near-ink observation weighting, coefficients clipped to ±4.
* Training: 8 families / 6 conservative lineages, 15 real weight instances, 30 including mild synthetic style variants. 698 supported exact sequences, 96x64 canvas, em=32, baseline y=45. Trained target identities are known across training families; this is missing appearance for known sequence content, not zero-shot Unicode learning.
* Selection: 48 fixed settings, only validation Regular instances of Baloo Chettan 2, Manjari, Uroob. Selected validation IoU 0.4056; clean best 0.4012.
* Holdout: Chilanka + Noto Sans/Serif Malayalam, 3 families / 2 groups / 7 weights. Macro IoU 0.3862; first learned model 0.3137; nearest-font baseline 0.4070. SDF MAE 0.0587, worse than first/retrieval. Three reference budgets, three permutations. Small correlated corpus; no writer-accuracy claim.
* Intended use: inspect local generated raster proposals from user-confirmed, exactly labelled, canonically aligned observed samples, with human review and generated provenance preserved.
* Unsupported: arbitrary unaligned crops; complete Malayalam repertoire; unseen target content; recovery of observed handwriting; automated export; font metrics/spacing/GSUB/GPOS; calibrated confidence; guaranteed readable conjuncts.
* Failure modes: contours fragment, missing strokes, averaged geometry, template-like bias, clipping limits, limited unusual-style generalization; low-resolution overlap does not establish readability.
* Data/rights: official Google Fonts/SMC commits and exact OFL notices retained. No personal handwriting/photos. No restricted research implementation. Original code MIT; font/dataset/model/output licensing remain separate. No unrestricted checkpoint distribution licence is asserted, and weights are not published.
* Provenance: `0165690674d47818bd66055cf66f7b8861bdbb557f964bea56154b33180ea045`; local `artifacts/best-validation.npz`, 61,793,153 bytes. See `manifests/artifacts.json` and [full campaign](reports/CAMPAIGN.md).
* Evaluation audit: known legacy-chillu aliases excluded from missing-target scores in protocol v2. Initial v1 reports are retained; v2 recomputation is not a new independent test set.

Every output must remain `provenance: model-generated`, `confirmed: false` initially, `not_observed_in_photo: true`, and `uncertainty: uncalibrated`. Human acceptance does not turn generated geometry into observed source handwriting. [Integration contract](docs/APP_INTEGRATION.md).
