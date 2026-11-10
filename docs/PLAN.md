# ImageCompressor reproduction implementation plan

> For agentic workers: implement inline with executing-plans and test-driven-development; review the whole result before completion.

**Goal:** A self-contained folder using the author's ImageCompressor, with native checkpoints, paper Algorithm 1 and quarter-budget step schedules.

**Architecture:** Vendor source from PreprocessingICM commit `6a77c4633cd5bb6750c77ea9d213d38ffc88ee7a`, keep the originals and a provenance manifest, and make only documented device/import compatibility changes. Reuse the tested BPG, dataset, training and evaluation infrastructure inside this folder; replace CompressAI-based proxy initialization and diagnostics with native-only equivalents.

**Spec:** The user's approved proposal and clarification: put the implementation in one folder, using the paper's ImageCompressor.

**Tech stack:** Python 3.10–3.12, PyTorch, TorchVision, NumPy/Pillow/SciPy, libbpg 0.9.8. CompressAI is not required.

## Global constraints

- All new work stays in this folder. Existing training jobs and `paper_fidelity` are untouched.
- Native ImageCompressor uses N=192/M=192 with context prediction enabled, and strict checkpoint loading.
- Proxy fine-tune: 100000 updates/QP, LR 1e-4 to 1e-5 after 80000. Fixed: 30000/20000. Adaptive: 25000/15000.
- Fine-tuning BPG requires a native RD-pretrained checkpoint; a separate explicit pretraining command is provided when none is available.
- No claim that a checkpoint from mbt2018 is an original/native checkpoint. No claims of completed training or paper-level BD-rate.
- Algorithm 1: real BPG forward/rate, proxy gradients, frozen task/proxy; fixed QP34 then adaptive QPs 28/31/34/37/41.

## Review focus

1. CPU/GPU noise and causal-mask behavior must match upstream mathematics.
2. Missing/mismatched native weights must fail, not silently load partial state.
3. Resume must retain Adam/sample position and enforce stage dependencies.
4. Paper text and demonstration configs differ; all choices and missing artifacts must be disclosed.
5. Kaggle packaging must include every local source and no credentials or existing account pool.

## Tasks

- [x] Task 1: vendor pinned proxy/NPP sources and implement native adapters. Tests: upstream parity, shape/gradient, strict checkpoint compatibility and rejection of CompressAI weights.
- [x] Task 2: implement native RD pretraining, BPG fine-tuning, calibration/audit, quarter schedules, deterministic crop manifest and NPP pipeline. Tests: native checkpoint training/resume, phase gates, real-value/proxy-gradient separation and LR boundary.
- [x] Task 3: provide runnable CLI, configuration, Kaggle packaging, Vietnamese README and provenance. Run complete local suite and a smoke pipeline; obtain an independent review and address material findings.

## Execution record

Ruling: use a new self-contained folder rather than a worktree, because the user explicitly requested a folder and the parent workspace contains unrelated intentional deletions.

Ruling: pretrained artifact availability is not assumed. Provide a functioning pretraining route and require explicit native artifacts for fine-tuning.

Task 1 complete: all 19 upstream blobs matched their Git blob SHA; native forward/rate/input gradients matched unmodified CPU upstream in train and eval, odd-size accounting passed, partial/mbt checkpoint rejection passed.

Ruling: keep the author's QA MLP initialization, without identity overrides. Supply normalized QP/51 as described by the paper; the released integration's lambda-based conditioning is a known code/text discrepancy and is documented.

Task 2 complete: 7/7 native/recipe unit tests passed. Real Windows libbpg integration smoke passed native RD pretraining 2+1 updates, Adam state restored at step 3, LR 1e-4 -> 1e-5 across the boundary, calibration, native BPG fine-tuning, fixed/adaptive transfer, real codec forward and frozen proxy/task with nonzero NPP gradients. Synthetic results are in docs/smoke.json and are not benchmarks.

Task 3 verification so far: complete unittest suite 9/9 passed; notebook and embedded runtime compile, recursive native source/QA modules/provenance included, weights/pool excluded. Vietnamese README and command template added. Independent final review pending.

Ruling: Kaggle builder emits explicit stage commands without submitting or selecting an undocumented RD pretraining budget; missing verified native weights are disclosed. User subsequently authorized publishing this folder to munnn01/image_compressor. Parent working-tree deletions stay untouched.

Final review: independent fresh-context review completed. Two important findings reproduced RED->GREEN: legacy native serialization support, and fixed->adaptive recipe/task/backbone/weight validation. Provenance inclusion in Kaggle was treated as required research traceability and fixed with a failing packaging test first. Final suite: 13/13 passed (`python -m unittest discover -s tests -v`). Real native/BPG smoke reran successfully after fixes. No remaining review findings deferred.

Final ruling on unverified behavior: GPU parity, full cloud execution, actual author pretrained artifact compatibility, detection-scale memory and scientific performance remain unverified and explicitly disclosed; these require the appropriate hardware, artifacts and completed experiments, not fabricated local evidence. Documented QP/51/padding/RD-budget/batch choices stand as implementation interpretations; they may change final performance and prevent claiming exact full-paper reproduction.

Task 3 complete: all nine example stage commands parsed; 19 original Git blobs and 18 generated source hashes verified. Publication targets a new empty repository through the GitHub connector, without modifying the parent Git index/HEAD/remotes. User authorization to push supplies the integration choice; no additional confirmation menu is needed.
