# Final independent review

A fresh-context read-only review checked native checkpoints, upstream mathematics/device allocation, optimizer/sample resume, stage dependencies, scientific claims and Kaggle source inclusion. No critical defect was reported. Two important findings were reproduced with failing tests and fixed:

- Native checkpoints saved in the legacy torch serialization format now retry without mmap, keeping `weights_only=True` and strict full architecture validation. This establishes format compatibility; the actual author pretrained artifact remains unavailable.
- Fixed training is restricted to QP34. Adaptive initialization requires a completed fixed checkpoint using real BPG level 8 and matching task, backbone and task weights. Wrong recipes are rejected before task model/data loading.

The missing provenance document in the embedded Kaggle archive was also fixed and covered by the packaging test, because source provenance is part of this research artifact's deliverable.

Verification: `python -m unittest discover -s tests -v` covers upstream CPU parity in train/eval, strict native checkpoints in current/legacy formats, CompressAI rejection, disjoint deterministic crops, phase gates and notebook completeness/compilation. `tools/smoke.py` covers native RD pretraining/resume, Adam/LR boundary, calibration, BPG fine-tuning and fixed/adaptive gradients with real Windows BPG and a synthetic frozen task.

Limits retained: GPU parity, full Kaggle execution, author pretrained weight compatibility, detection-scale memory and scientific performance have not been established. QP/51 conditioning, padding, RD budget, task/batch choices and same-path Kaggle resume are documented interpretations/constraints. None is represented as a verified identical full-paper result.
