# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RetailGaze is a deep learning research project for retail customer gaze estimation — predicting where customers are looking in a retail scene. The primary contribution is `Face3D`, a model that combines monocular depth estimation (MiDaS DPT_Hybrid) with face-based gaze prediction using a modified ResNet-50 backbone.

The dataset is **RetailGaze** (custom, pickle-serialized), stored externally at paths like `/media/primesh/F4D0EA80D0EA49061/PROJECTS/FYP/Gaze detection/Datasets/RetailGaze (our)/`. Dataset files are not in this repo; paths in scripts must be updated per machine.

## Running Training and Testing

Training and testing are run as Python scripts directly (no build system):

```bash
# Train Face3D model
python test_face3d.py

# Run base model test
python test_base.py
```

Notebooks (`.ipynb`) are used for prototyping (e.g., `train_base.ipynb`). The `Gaze_Dual_Attn/` submodule contains separate Jupyter notebooks (`Dual_Attn.ipynb`, `Other_Models.ipynb`) for evaluating dual-attention baselines.

TensorBoard logs are written to `runs/`:
```bash
tensorboard --logdir=runs/
```

Training loss CSVs are written to paths hardcoded in training scripts (some point to `/content/drive/MyDrive/` for Colab use).

## Architecture

The codebase follows a consistent three-layer structure:

| Layer | Directory | Purpose |
|---|---|---|
| Models | `models/` | PyTorch `nn.Module` definitions |
| Dataloaders | `dataloader/` | `torch.utils.data.Dataset` subclasses |
| Training loops | `training/` | Train/test functions and optimizer wrappers |

**Key models:**
- `models/face3d.py` — `Face3D`: scene image → MiDaS depth map (frozen) + face crop → ResNet-50 → 3D gaze vector `(dx, dy, dz)`. The depth branch is always frozen (`torch.no_grad()`).
- `models/dual_path.py` — `GazeDual`: two-stream resnest50 (scene + head crop) → feature concat → deconv upsampling → heatmap.
- `models/hypo_attention.py` — `HypoAttention`: VGG16 dual-stream with attention gating + 5-shifted-grid output heads (outputs 5 probability distributions over a 5×5 grid, shifted in cardinal directions).
- `models/base.py` — `BaseModel`: simple ResNet-50 → 2D gaze point (baseline).
- `models/hypo_attention_c2f.py` — coarse-to-fine variant of HypoAttention.

**Key datasets in `dataloader/face3d.py`:**
- `RetailGaze` — primary dataset class; pickle-based, 640×480 images, returns `(img, face, head, gt_label, head_box, image_path)`. Scene images are resized to 224×224; face crops to 448×448.
- `GooDataset` — for the GOO (Gaze in the Open) dataset using the same pickle format.
- `GazeDataset` — for Gaze360 / GazeFollow using `.mat`/`.txt` annotation files.

**Label conventions:** gaze targets are normalized `(x/640, y/480)`. Head boxes come from `data['ann']['hbox']`. The `Face3D` training loop constructs 3D labels on-the-fly by computing relative depth between head and gaze target using the MiDaS output.

**Evaluation metric:** angular error (degrees) computed via dot-product cosine similarity, plus L2 Euclidean distance for 2D predictions.

## Key Dependencies

- PyTorch + torchvision
- `resnest` (resnest50 backbone, install via `pip install resnest`)
- MiDaS loaded via `torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")` — requires internet on first run
- `tensorboard`, `tqdm`, `scikit-learn`, `opencv-python`, `scipy`, `Pillow`
- `early_stopping_pytorch/pytorchtools.py` — bundled in-repo

## Checkpoints

Model checkpoints are saved/loaded from external paths (e.g., `/media/primesh/.../saved_weights/`). The `EarlyStopping` class in `early_stopping_pytorch/pytorchtools.py` saves `checkpoint.pt` to the working directory by default. Checkpoint dict keys: `state_dict`, `optimizer`, `epoch`.

## Gaze_Dual_Attn Submodule

`Gaze_Dual_Attn/` is a self-contained evaluation environment (has its own `models/`, `dataloader/`, `training/`, `utils.py`). It is notebook-driven and mirrors the root structure. Changes to root-level modules do not propagate there automatically.
