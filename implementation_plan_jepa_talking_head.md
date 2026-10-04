# Implementation Plan: Talking Head JEPA Prototype

This plan outlines the architecture and implementation strategy for a Joint-Embedding Predictive Architecture (JEPA) tailored for audio-driven talking head synthesis, inspired by datasets and pipelines like SyncTalk. 

## User Review Required

> [!NOTE]
> Please review the architecture proposed below. The plan assumes we are taking pre-extracted audio features (e.g., from HuBERT or SyncTalk's Audio-Visual Encoder) and video frames of faces as inputs. Let me know if you would prefer to process raw waveforms or 3D blendshape coefficients instead of images.

## Open Questions

> [!IMPORTANT]
> 1. Should the `FaceEncoder` use a pre-trained Vision Transformer (ViT), or a lighter CNN like ResNet for faster prototyping?
> 2. SyncTalk often processes 512x512 images. Should the prototype be configured for 512x512, or downsampled to 224x224 for standard ViT compatibility?

## Proposed Architecture: `talking_head_jepa.py`

We will create a new standalone file `talking_head_jepa.py` containing the prototype classes.

### 1. `AudioFeatureEmbedder` (Action Encoder)
Instead of arbitrary "actions", the model is driven by audio. This module will take pre-extracted audio features (e.g., shape `(Batch, Time, AudioFeatureDim)`) and project them into the conditional embedding space used by the autoregressive predictor.
- Architecture: 1D Convolutions or MLP to map sequence of audio features to sequence of condition embeddings.

### 2. `FaceEncoder` (Vision Encoder)
This module encodes the ground-truth facial video frames into latent state representations.
- Architecture: ViT (Vision Transformer) or a convolutional backbone.
- Input: Context and Target video frames `(Batch, Time, Channels, Height, Width)`.
- Output: State embeddings `(Batch, Time, Dim)`.

### 3. `TalkingHeadJEPA` (Core Model)
The main module coordinating the prediction.
- **`encode()`**: Processes the `face_frames` using `FaceEncoder` and `audio_features` using `AudioFeatureEmbedder`.
- **`predict()`**: Uses an autoregressive transformer (like `ARPredictor` from `module.py`) to predict future facial latent states. It takes the past facial state embeddings (context) and the *full* sequence of audio embeddings (context + future) to predict the future facial state embeddings.
- **`forward()` (Training Step)**:
  - Takes a batch containing a sequence of face frames and a sequence of audio features.
  - Slices the frames into `context_frames` and `target_frames`.
  - Encodes the frames to get `context_embs` and `target_embs`.
  - Encodes the audio features.
  - Autoregressively predicts the `target_embs`.
  - Computes the Mean Squared Error (MSE) loss between the predicted embeddings and actual `target_embs`.
  - Applies Sketch Isotropic Gaussian Regularization (`SIGReg`) to the state embeddings to prevent representation collapse.

## Verification Plan

### Automated Tests
- I will run `python -m py_compile talking_head_jepa.py` to ensure no syntax errors.
- I will write a small `__main__` block at the bottom of the file to instantiate the model with dummy tensors (e.g., random audio features and video frames) and run a forward pass to verify tensor shapes and loss computation.

### Manual Verification
- You can inspect the newly created `talking_head_jepa.py` file to ensure the architecture aligns with your vision for the SyncTalk adaptation.
