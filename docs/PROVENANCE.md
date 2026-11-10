# Provenance and compatibility changes

Upstream: XingtongGe/PreprocessingICM, commit `6a77c4633cd5bb6750c77ea9d213d38ffc88ee7a`, Apache-2.0. Exact originals are stored in `upstream/`. The manifest records Git blob IDs and SHA256 for 19 downloaded files. `tools/build_vendor.py` produces the package-relative port deterministically and records output hashes in `upstream/port_manifest.json`.

Changes to vendored modules:

1. Package-relative imports, so ImageCompression's top-level `models` does not collide with NPP or unrelated packages.
2. Quantization noise allocated with the input device/dtype; upstream CPU noise allocation would fail on CUDA.
3. Removed diagnostic prints.
4. Causal convolution uses `forward` with `F.conv2d(weight * mask, ...)`, avoiding `.data` mutation and preserving masking in parameter gradients. Forward values and input derivatives match upstream exactly on the CPU parity fixture. Parameters under the causal mask now have zero gradient explicitly.

Architecture/math retained: N=M=192, analysis/synthesis, GDN, hyperprior, context predictor, sigma squaring/clamping, Gaussian CDF mass for y and learned BitEstimator for z, original likelihood stabilizers/rate clamps, additive noise in training, rounding in evaluation, and unclipped MSE for native RD training.

Adapter changes: replicate-pad proxy inputs to multiples of 64 and NPP inputs to multiples of 16, crop output back to original geometry and count padded proxy bits per original pixel. Padding is unnecessary for 256×256 proxy crops but is needed for 224×224 classification task crops. This adapter behavior is tested separately from upstream parity.

NPP wrappers keep the author's fixed/adaptive modules and default initialization. Fixed parameters transfer strictly into adaptive; QA MLP values remain those initialized by the author module. The input is normalized QP/51. Public integration uses lambda conditioning, so this is a documented interpretation of paper text, not a proven identical training recipe.

The BPG/task/CLI/checkpoint infrastructure was adapted from the workspace's `paper_fidelity` implementation. No legacy weights are imported. Native checkpoint loading is complete and strict, and the stage gate requires native metadata, correct QP/codec, complete quarter-budget steps and step schedules before NPP training. These checks establish structural compatibility and recipe metadata, not authenticity of external raw state_dict provenance.

No verified author pretrained weights are bundled. Native RD pretraining is available with an explicit budget. Synthetic smoke checkpoints are temporary and are deleted after the smoke run; smoke metrics are not scientific results.
