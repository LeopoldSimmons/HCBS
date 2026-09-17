# HCBS maintenance interface

This change has not been tested, compiled, trained, or benchmarked, as requested.
The paper's published numbers are historical results, not measurements of this
revision. No new trained weights, labels, or experimental partitions are fabricated.

## Environment and source integration

Use Python 3.10 or 3.11 and install `requirements.txt`. The pinned environment is
a candidate configuration, not a validated compatibility claim. Install matching
torch/torchvision CUDA wheels for your machine. The default DCN convolution uses
`torchvision.ops.deform_conv2d`, preserves parameter names and the existing offset
and mask layout, and does not compile the legacy extension. Numerical equivalence
and backward correctness have not been established by this maintenance change.
`--dcn_backend legacy` retains the historical compiled extension path; it needs its
own historical environment. Deformable ROI pooling remains legacy-only and is not
used by the detector. Deterministic mode may reject unsupported CUDA operations.

The HCBS source can run directly from `Project/src`. Obtain the matching MOC
dataset GT, RGB frames and COCO/ImageNet initialization separately. If integrating
into a MOC checkout, record its exact commit and copy directory contents:

```bash
git -C MOC-Detector rev-parse HEAD
cp -a Project/src/. MOC-Detector/src/
```

The authentic upstream experiment commit is not known from this repository, so it
is deliberately not guessed. Default COCO weights are still expected under
`../experiment/modelzoo/` relative to `Project/src`. Alternatively pass
`--pretrain_model imagenet`, which downloads the original backbone initialization.

## Dataset scope and partitions

`--data_root` (or `HCBS_DATA_ROOT`) applies to all three datasets. Expected folders:
`UCF101_v2`, `JHMDB`, `multisports`. Each contains its original GT pickle and
`rgb-images/<video-id>/<frame>.(jpg|png)`; text features are at
`numpys/<video-id>/<frame>.npy` or under `--text_root`.

Defaults retain the repository's one-class UCF/J-HMDB and 15-class MultiSports
heads. `--num_classes` must agree with the GT label list and zero-based class IDs.
This does not filter or remap full-dataset annotations automatically. Supply the
actual subset GT or a correctly configured full-dataset experiment.

Provide a JSON manifest with `dataset`, `split`, `train`, `val`, `test`; see
`Project/configs/split.example.json`. Replace every placeholder with a real GT
video ID. Lists must be disjoint and internally unique. Unlisted videos are
allowed for an explicit subset experiment. Without a manifest the original train
and test lists remain available, but validation and best-model selection are
disabled. Test data never substitutes for a missing validation partition.

Archive inventory from the earlier audit (not rerun for this change):

| Archive | Text files | Video directories with text | Observed class |
| --- | ---: | ---: | --- |
| J-HMDB | 1,129 | 43 | kick_ball |
| UCF101-24 | 43,356 | 147 | SoccerJuggling |
| Multisports | 77,420 | 127 | no class directory level |

The relation to the paper's 712 clips, all subset definitions, original partitions,
and baseline comparison protocols still require an author-supplied manifest.
Video directory count is not assumed to equal the paper's clip count.

## Text feature contract

The default `--text_format feature_map` reads float32 `[64,H/4,W/4]` arrays (also
accepts a leading singleton batch dimension) and passes them directly to the
fusion branch. They do not traverse the three-channel image stem or update image
BatchNorm. The provided projection exporter produces `[64,72,72]`, requiring
288x288 detector input. Other resolutions require matching features; no silent
resizing, channel guessing or random projection is performed.

`--text_format legacy_image` explicitly retains `[3,H,W]` shared-backbone input
and accepts legacy `[H,W]` arrays by channel replication. This mode retains shared
BatchNorm behavior and is not claimed equivalent to feature_map mode. Neither
feature format is asserted to reproduce the original paper without matched weights.
Text maps represent semantic features, not image pixel locations: flipping the
video duplicates the semantic input and does not spatially flip the embedding.

The zero-target text objective is disabled. Export only an existing, complete,
author-verified BERT checkpoint with `hcbs_text_projection.pt`:

```bash
python Project/BERT_txt_feature.py \
  --sentence_root /data/descriptions \
  --model_path /models/verified-bert-projection \
  --output_root /data/hcbs-features-v1
```

The output root must be empty. Relative video/frame paths are preserved; the
exporter writes feature and source hashes in `features.json`. Keep an exported
feature revision immutable. Legacy features without this manifest require an
explicit `--text_revision` whose identity you must change whenever content changes.
Never fit the text encoder on validation/test descriptions. The original large
projection is retained for checkpoint compatibility; replacing it with a compact
encoder or a new learning objective is a separate research change, not a verified
maintenance fix. A full reproducible MLLM description-generation runner and the
original dialogue/model/sampling configuration remain author-supplied prerequisites.

## Training and resume

Run from `Project/src` with actual paths:

```bash
python train.py --dataset ucf101 --gpus 0 --batch_size 4 \
  --data_root /data --split_manifest /configs/split.json \
  --modality multimodal --text_format feature_map \
  --text_root /data/hcbs-features-v1 --save_dir /runs/hcbs \
  --val_epoch --auto_stop
```

Use `--modality visual_only` for RGB baselines or optical flow (`--ninput 5`).
ResNet supports visual-only mode. Optimizers are created after input-channel
conversion. `--auto_stop` is retained as a compatibility option name but now means
validation-AP checkpoint selection using normal inference in the same modality.
It does not perform early stopping, test selection or optimizer rollback to best
weights. The epoch LR schedule is saved and restored. Even an initial AP of zero
saves a best checkpoint; non-finite AP fails explicitly.

`--load_model PATH` loads strict weights for finetuning; `--resume --load_model PATH`
restores optimizer, epoch, best AP, LR schedule and RNG states. Resume uses a trusted
training checkpoint. Its dataset, partitions, modality, text revision, architecture
and LR schedule must match. `--ucf_pretrain` allows only heatmap/movement heads to
be reinitialized and starts a new optimizer. `--allow_legacy_checkpoint` accepts
missing protocol metadata but never silently fills missing model parameters.
If resuming in a new output directory, preserve the earlier `model_best.pth`
separately: it is not embedded inside `model_last.pth`.

## Inference, evaluation and caches

```bash
python det.py --task normal --dataset ucf101 --gpus 0 --batch_size 4 \
  --data_root /data --split_manifest /configs/split.json --eval_split test \
  --modality multimodal --text_format feature_map \
  --text_root /data/hcbs-features-v1 --rgb_model /runs/hcbs/model_best.pth \
  --inference_dir /runs/hcbs/predictions
```

Each run prints its full output directory, identified by model weights, source,
GT, partition, text revision, modality and inference configuration. Use that exact
directory for evaluation and repeat the same dataset/protocol options:

```bash
python ACT.py --task frameAP --dataset ucf101 --data_root /data \
  --split_manifest /configs/split.json --eval_split test \
  --modality multimodal --inference_dir /runs/hcbs/predictions/PRINTED_HASH
```

Run `--task BuildTubes` and then `--task videoAP` with the same protocol for video AP.
`--redo` regenerates selected outputs; it never deletes unrelated cache directories.
Existing old caches are preserved; evaluation requires `--allow_legacy_predictions`
to accept missing provenance. Do not mutate input frames, GT or feature exports in
place: create a new dataset/feature revision and output root instead.

`--task stream` and `--task speed_test` require `--modality visual_only`. These paths
do not represent full HCBS multimodal inference. Synthetic detector FPS excludes
video decoding, MLLM description generation and text encoding. Visualization creates
a new per-run directory, requires an explicit class-label JSON list, preserves raw
outputs and never clears a shared `tmp` folder.

## Validation status

No tests or verification were run for this change at the user's request. Existing
syntax CI is unchanged; it is not evidence of runtime or numerical correctness.
Before making new benchmark claims, establish forward/backward correctness, matched
checkpoint inference, AP edge cases, deterministic resume, multi-GPU tails, DCN
equivalence, subset provenance and the intended semantic training objective.
