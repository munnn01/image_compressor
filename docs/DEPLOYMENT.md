# Native RD deployment — 10 November 2026

The user approved a separate native RD budget of 100,000 optimizer updates per QP, with Adam LR 1e-4 through update 80,000 then 1e-5. The user also explicitly requested using multiple owned accounts. Official API authentication and accelerator quota were checked before allocation.

| QP | Account | Public notebook |
|---|---|---|
| 28 | nguoingoaihanhlang | [Native RD QP28](https://www.kaggle.com/code/nguoingoaihanhlang/imagecompressor-native-rd-qp28) |
| 31 | nguyenhoanglan1232 | [Native RD QP31](https://www.kaggle.com/code/nguyenhoanglan1232/imagecompressor-native-rd-qp31) |
| 34 | thuha1205 | [Native RD QP34](https://www.kaggle.com/code/thuha1205/imagecompressor-native-rd-qp34) |
| 37 | tranthihongdieu | [Native RD QP37](https://www.kaggle.com/code/tranthihongdieu/imagecompressor-native-rd-qp37) |
| 41 | htran123456 | [Native RD QP41](https://www.kaggle.com/code/htran123456/imagecompressor-native-rd-qp41) |

All five initial versions were accepted and reported RUNNING by the official API. This is a deployment-time snapshot, not a live status or proof of completed training. Each session targets up to 16,384 additional updates, retains the full 100,000 target and saves every 256 updates. Local controllers validate the completed session's native provenance, full schedule and exact step increment, then attach that version's outputs for continuation on the same account. They stop on a failed or rejected submission; no account changes happen automatically. Continuation requires the local computer/controller to remain running.

The earlier all-QP native bank notebook was superseded only after all five replacements reported RUNNING; cancellation was requested for its observed version-1 session. Existing mbt2018 comparison notebooks are separate and were not cancelled by this deployment.

Architecture: pinned author ImageCompressor(192,192); initial native RD weights are trained from scratch. Data: eeshawn/flickr30k/1, 800,000 fixed virtual 256×256 crops, seed 2024, 64 source images held out. Batch=4. The RD notebooks do not use BPG or any external codec checkpoint. Each executes a CUDA native forward/input-gradient smoke before training. Actual smoke results and progress must be read from job outputs; no paper-level metric is claimed.

RD pretraining is followed by native calibration/audit, then BPG fine-tuning 100,000/QP, fixed NPP 30,000 and adaptive NPP 25,000. These subsequent stages have not been launched by the initial RD submission.

## BPG phase continuation

After all five RD jobs completed 100,000 updates, the user requested continuing the native direction. Five new notebooks use the same accounts and names `imagecompressor-native-bpg-qp28`, `...qp31`, `...qp34`, `...qp37`, `...qp41`. Each attaches all five immutable RD version-7 outputs, verifies native architecture/stage/schedule/crop provenance, calibrates on the 64 held-out source images against real BPG JCTVC level 8, then fine-tunes its QP from the selected native RD checkpoint.

The BPG phase has its own 100,000-update target per QP, with LR 1e-4 through update 80,000 then 1e-5. Sessions add at most 8,192 updates at batch 4, saving every 256 updates. Each ends with a real-BPG holdout audit. Resume restores the BPG optimizer; it does not restart from RD weights or redo checkpoint selection. Local continuation controllers require the computer to remain running and stop on failed/rejected jobs.

Native proxy builder: `kaggle/prepare_proxy.py`. Its `--rd-sources` input is a JSON mapping of all five QPs to their version-pinned native RD notebook references. The deployed full-budget BPG fine-tuning is still in progress; no new task accuracy/BD-rate result is claimed. Fixed/adaptive NPP remain subsequent stages.

Initial API snapshot: QP28/31/34/41 RUNNING; QP37 queued on tranthihongdieu, with a second authorized trial on shungg05 also queued. A local coordinator keeps the first trial that reaches RUNNING, cancels the other observed session after that condition holds, and starts continuation for the winner. This avoids committing to duplicate training while GPU workers are unavailable. Statuses are historical deployment snapshots; check Kaggle for current state.

## GPU selection correction

The user briefly requested avoiding T4, then explicitly approved T4 after the hardware check. Kaggle retired P100 on 15 September 2026 and automatically maps P100 requests to T4. The attempted P100-labelled jobs were therefore not P100 runs. Current builders and continuation commands explicitly select NvidiaTeslaT4 (T4 x2); actual hardware should be read from the runtime/UI, not inferred from notebook names.

The active QP28/31/34/41 notebook slugs retain `-p100` from that unsuccessful selection attempt, and QP37 retains `-retry`. These names are historical only. Current remote jobs are retained; changing the local controllers does not cancel or restart them.

During the earlier switch attempt, the four original native T4 sessions were cancelled. No complete BPG checkpoint could be retrieved (local download attempts produced empty checkpoint files), so those four replacements initialize from the completed native RD bank and recalibrate. Unpublished BPG updates from interrupted sessions are not claimed preserved. The total BPG target remains 100,000 updates/QP with decay after 80,000. Completed RD pretraining is preserved.

Official notice: https://www.kaggle.com/product-announcements/735239

The initial QP37 BPG run failed before calibration/training because adding notebook inputs changed the absolute dataset mount path, changing the newly generated manifest hash. The fix restores the exact RD manifest from a pinned RD output and uses `--image-root` for the current dataset mount. Crop membership, holdout split, seed and hash remain unchanged. QP37 version 2 was accepted and reported RUNNING on explicit T4. Other initial jobs encountering this exact error are eligible for one retry with the corrected builder; unrelated errors are not automatically retried.
