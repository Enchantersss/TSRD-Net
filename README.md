# TSRD-Net: Thin-Structure Reconstruction Decoder Network

This repository contains the initial PyTorch code release for **TSRD-Net**, a decoder-centered architecture for crop organ segmentation in field images.

**Paper:** *A Thin-Structure Reconstruction Decoder Network for Crop Organ Segmentation in Field Images*

TSRD-Net targets thin, curved, sparse, and partially occluded wheat stems. On the GWFSS competition dataset, the architecture alone achieves **66.28% mIoU** and **35.41% Stem IoU**. With targeted CutMix and a composite loss during training and CRFasRNN refinement at inference, it achieves **68.17% mIoU** and **39.34% Stem IoU**.

## Method Overview

TSRD-Net reconstructs high-resolution features with three decoder components:

- **Upsampling Transformer (UpTR):** uses windowed cross-attention to reconstruct high-resolution decoder features.
- **Hybrid Spatial Interactive Attention (HybridSIA):** combines local and global information to refine the sum of upsampled decoder features and filtered skip features.
- **Dynamic Channel Reconstruction (DCR):** uses selective deformable branches at the highest decoder resolution to refine curved, thin structures.

Targeted CutMix and the composite loss are training choices. CRFasRNN is an inference-time refinement step. Their effects are reported separately from the decoder architecture below.

## Main Results

All experimental values below are taken directly from the manuscript. Segmentation scores are percentages.

### GWFSS Competition Dataset

The task distinguishes background, head, stem, and leaf. The network-only and full-configuration results are listed separately.

| Model | Params (M) | mIoU | mF1 | Background IoU | Head IoU | Stem IoU | Leaf IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PSPNet | 46.58 | 61.16 | 72.72 | 65.49 | 79.97 | 21.55 | 77.64 |
| SegFormer | 3.67 | 60.11 | 71.05 | **68.76** | 78.33 | 16.09 | 77.24 |
| K-Net | 25.15 | 59.23 | 70.43 | 67.65 | 79.66 | 15.93 | 73.68 |
| MSEONet | 44.56 | 54.07 | 65.50 | 61.31 | 72.37 | 9.69 | 72.89 |
| U-Net | 28.25 | 54.00 | 64.69 | 62.57 | 72.60 | 6.64 | 74.20 |
| Fast-SCNN | 1.50 | 51.70 | 62.94 | 60.23 | 67.67 | 6.19 | 72.72 |
| SSRS | 27.12 | 50.50 | 63.42 | 55.61 | 65.39 | 12.08 | 68.93 |
| Swin-UMamba | 59.89 | 50.83 | 64.37 | 58.80 | 58.60 | 15.55 | 70.35 |
| GeoSA-BaSA | 92.78 | 39.96 | 53.82 | 49.93 | 42.88 | 7.58 | 59.44 |
| UrbanSSF | 59.95 | 64.89 | 76.55 | 65.89 | 84.53 | 31.20 | 77.93 |
| TSRD-Net network only | 56.97 | 66.28 | 77.99 | 67.39 | 84.41 | 35.41 | 77.91 |
| TSRD-Net + CutMix + composite loss + CRFasRNN | 56.97 | **68.17** | **79.62** | 68.74 | **85.13** | **39.34** | **79.45** |

Parameter counts include frozen parameters. PSPNet through GeoSA-BaSA use the cross-dataset implementations configured for four classes; UrbanSSF and both TSRD-Net variants use the GWFSS configurations.

### GWFSS Network Structure Ablation

Baseline is the matched UrbanSSF implementation. Each row adds the listed modules to Baseline under the same base training recipe.

| Setting | mIoU | Background IoU | Head IoU | Stem IoU | Leaf IoU | mF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 64.89 | 65.89 | 84.53 | 31.20 | 77.93 | 76.55 |
| + HybridSIA | 65.10 | 66.24 | **84.81** | 31.65 | 77.70 | 76.75 |
| + UpTR | 65.75 | 66.98 | 84.47 | 33.64 | 77.91 | 77.43 |
| + DCR | 65.28 | 66.05 | 84.10 | 33.30 | 77.67 | 77.08 |
| + HybridSIA + UpTR | 65.94 | 67.15 | 84.74 | 33.87 | **78.00** | 77.58 |
| + UpTR + DCR | 66.12 | 67.29 | 84.46 | 35.01 | 77.72 | 77.84 |
| + HybridSIA + DCR | 65.46 | 66.47 | 84.56 | 33.35 | 77.46 | 77.20 |
| + HybridSIA + UpTR + DCR (TSRD-Net) | **66.28** | **67.39** | 84.41 | **35.41** | 77.91 | **77.99** |

### Architecture, Training, and Inference Ablation

Each individual setting is relative to Baseline. The final row combines the decoder architecture, both training changes, and inference-time refinement.

| Setting | mIoU | Background IoU | Head IoU | Stem IoU | Leaf IoU | mF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 64.89 | 65.89 | 84.53 | 31.20 | 77.93 | 76.55 |
| TSRD-Net architecture only | 66.28 | 67.39 | 84.41 | 35.41 | 77.91 | 77.99 |
| CutMix only | 65.73 | 68.18 | 84.73 | 32.16 | 77.85 | 77.26 |
| Composite loss only | 65.76 | 65.41 | 85.03 | 34.58 | 78.03 | 77.51 |
| CRFasRNN only | 65.57 | 66.25 | 84.68 | 33.24 | 78.12 | 77.25 |
| Architecture + CutMix + composite loss + CRFasRNN | **68.17** | **68.74** | **85.13** | **39.34** | **79.45** | **79.62** |

### Generalization: Sugar Beets 2016

Models were trained separately on Sugar Beets 2016 to distinguish background, sugar beet, and weed. The temporal split uses 200 training frames (23–222), leaves frames 223–245 unused, and uses 60 validation frames (246–305). All models have a 100-epoch training budget; each row reports the epoch with the highest validation mIoU. This evaluates the architecture on a second agricultural task with labeled training data.

| Model | Best epoch | mIoU | mF1 | OA | Background IoU | Sugar beet IoU | Weed IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TSRD-Net | 89 | **81.77** | **88.56** | 99.18 | 99.32 | 91.78 | **54.20** |
| Swin-UMamba | 69 | 81.74 | 88.49 | **99.22** | **99.35** | **92.16** | 53.69 |
| U-Net | 90 | 80.63 | 87.64 | 99.11 | 99.25 | 91.33 | 51.32 |
| UrbanSSF | 98 | 80.60 | 87.64 | 99.08 | 99.22 | 91.11 | 51.48 |
| MSEONet | 82 | 79.43 | 86.71 | 99.02 | 99.18 | 90.22 | 48.89 |
| SSRS | 96 | 79.16 | 86.48 | 98.99 | 99.14 | 90.08 | 48.25 |
| K-Net | 73 | 78.57 | 85.98 | 98.92 | 99.02 | 89.90 | 46.78 |
| GeoSA-BaSA | 82 | 78.45 | 85.89 | 98.96 | 99.08 | 89.69 | 46.58 |
| SegFormer-B0 | 69 | 73.40 | 81.39 | 98.51 | 98.58 | 86.25 | 35.37 |
| Fast-SCNN | 65 | 73.37 | 81.37 | 98.41 | 98.52 | 86.21 | 35.38 |
| PSPNet | 79 | 61.41 | 70.72 | 96.49 | 96.41 | 68.08 | 19.75 |

Bold entries indicate the highest segmentation score in each column.

## Code

The model source is provided in [`tsrd_net.py`](tsrd_net.py). This repository is an initial code release.