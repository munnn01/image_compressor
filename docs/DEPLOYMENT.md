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
