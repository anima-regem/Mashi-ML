# Separate licensing layers

* Original Mashi-ML source code: MIT, as stated in the root LICENSE.
* Source fonts: exact notices in `fonts/`, asset provenance/hashes in `manifests/corpus.lock.json`. Use OFL-1.1 where offered, including explicit OFL alternatives in dual-licensed sources. Preserve complete notices, including any additional licence text.
* Acquired binaries and rendered training data: local only, excluded from Git. No newly assigned blanket dataset licence overrides font rights. Reproduce from the pinned acquisition script. Original fonts can only be redistributed subject to their own notices and conditions.
* Trained checkpoints: local research artifacts, hashed and documented. This campaign does not assert that OFL automatically supplies a licence for ML-weight distribution; no checkpoint has been published or assigned MIT/OFL by inference.
* Generated font geometry: preserve lineage. Font derivative conditions and reserved-name obligations need review before distribution; no automatic unrestricted-output claim.

The manifest records an explicit per-source usage decision and exact licence header. No project-specific Reserved Font Name declaration was found before PREAMBLE in the acquired notices. The full notices and embedded font name records should still be checked before exporting a derivative. Never use an upstream family name for a generated Mashi family.

The excluded-source record explains GPL-only sources and unavailable RIT binaries. No noncommercial research implementation was imported. The model is an original NumPy/SciPy implementation, not MX-Font, LF-Font or FUNIT code.

SIL [FAQ 1.25–1.26](https://openfontlicense.org/ofl-faq/) treats AI-generated fonts based on OFL inputs as derivatives retaining OFL, while distinguishing non-font graphical output. This is a documented font-output condition, not an explicit prohibition on training or all checkpoint distribution. Checkpoint classification/packaging remains unresolved here, especially because this file contains 15 recoverable source-font SDF templates in addition to learned arrays. Withholding weights is this project's conservative release decision, not a demonstrated licence ban. See the [audit](../reports/AUDIT.md).

`archive/pilot/` preserves the earlier Mashi research pilot supplied by the application task, with its recorded source manifest/results. Its training inventory, normalization and tiny corpus differ from this campaign. The pilot checkpoint remains local. See that folder's provenance note.
