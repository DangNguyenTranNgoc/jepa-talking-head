import os
import urllib.request

import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

# Import components from the existing Le-WM codebase
from jepa import JEPA
from module import ARPredictor, MLP, SIGReg

class AudioFeatureEmbedder(nn.Module):
    """
    Embeds raw audio features (e.g., from HuBERT or SyncTalk's Audio-Visual Encoder)
    into the conditional embedding space used by the AR Predictor.
    """
    def __init__(self, audio_feature_dim, emb_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, audio_features):
        """
        Args:
            audio_features (Tensor): (Batch, Time, AudioFeatureDim)
        Returns:
            Tensor: Action embeddings for JEPA (Batch, Time, EmbDim)
        """
        return self.net(audio_features)


class FaceEncoder(nn.Module):
    """
    Vision encoder wrapper for facial frames.
    Uses a standard PyTorch Vision Transformer (ViT) to encode 
    224x224 facial frames into latent representations.
    """
    def __init__(self, image_size=224, embed_dim=768, pretrained_imagenet=False):
        super().__init__()
        # pretrained_imagenet=False (default) preserves the original from-scratch
        # prototyping behavior. pretrained_imagenet=True initializes from
        # torchvision's supervised ImageNet1K weights before any JEPA training -
        # added 2026-08-20 after probing found the from-scratch encoder
        # underperforming a frozen, never-JEPA-trained ImageNet ViT baseline on
        # facial-landmark probing. This lets a fine-tuning run isolate "does
        # JEPA's predictive objective help on top of a strong backbone" from
        # "ViTs need a lot of pretraining data to work well," which the
        # from-scratch setup couldn't distinguish.
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained_imagenet else None
        self.vit = vit_b_16(weights=weights)
        self.vit.heads = nn.Identity() # Remove the classification head
        self.embed_dim = embed_dim
        self.pretrained_imagenet = pretrained_imagenet
        
    def forward(self, pixels, interpolate_pos_encoding=True):
        """
        Args:
            pixels (Tensor): Flattened batch of frames (Batch * Time, Channels, Height, Width)
        Returns:
            A struct/object with .last_hidden_state containing the sequence with the CLS token.
            This mimics the HuggingFace ViT output format expected by jepa.py.
        """
        # Process input through ViT's patch embedding
        x = self.vit._process_input(pixels)
        n = x.shape[0]
        
        # Append class token
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Pass through the transformer encoder
        x = self.vit.encoder(x)
        
        # Wrap output to match the expected format in jepa.py: output.last_hidden_state
        class Output:
            pass
        out = Output()
        out.last_hidden_state = x
        return out


class FarlFrozenFaceEncoder(nn.Module):
    """
    Frozen FaRL (CVPR 2022, `FacePerceiver/FaRL`, face-specific CLIP ViT-B/16)
    visual encoder, added 2026-08-26 after JEPA_Latent_Probing.ipynb's Part
    2g/2i found this exact representation (`farl_patch_raw`: dense,
    pre-projection ViT patch tokens, mean-pooled - NOT CLIP's pooled
    image-text `encode_image()` embedding, which ties the generic ImageNet
    ViT baseline rather than beating it) is the strongest single-identity
    representation this project has produced, clearly beating both a
    generic ImageNet ViT-B/16 and this project's own best fine-tuned JEPA
    checkpoint (~1.69 vs ~1.87/~1.81 NME on landmark probing) - with zero
    fine-tuning. Always frozen: `requires_grad_(False)` on every parameter,
    and `.train()` is overridden to force eval mode permanently, regardless
    of what the outer `TalkingHeadJEPA.train()` call does. Unlike
    `FaceEncoder`, there is no `pretrained_imagenet`-style fine-tuning
    question here - only `projector`/`predictor`/`audio_embedder`/
    `pred_proj` train on top of this encoder's fixed output.

    Cross-identity note (2026-08-26, Part 5): this representation's
    single-identity advantage does NOT transfer to held-out identities
    under a linear/mean-shift probe - same hard floor every other
    representation in this project hit. Fine for this project's actual
    scope (single-identity `May`); would need revisiting before ever using
    this encoder on an identity outside the training set.

    Requires the `clip` package (`pip install ftfy regex && pip install
    git+https://github.com/openai/CLIP.git`) - not a hard dependency of the
    rest of this module, only imported inside `__init__` so `face_encoder_type='vit_b16'`
    (the default) never needs it installed.
    """

    FARL_CHECKPOINT_URL = (
        "https://github.com/FacePerceiver/FaRL/releases/download/"
        "pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep16.pth"
    )

    def __init__(self, embed_dim=768, checkpoint_dir="probing_cache/farl"):
        super().__init__()
        import clip  # local import - optional dependency, see class docstring

        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, "FaRL-Base-Patch16-LAIONFace20M-ep16.pth")
        if not os.path.isfile(ckpt_path):
            urllib.request.urlretrieve(self.FARL_CHECKPOINT_URL, ckpt_path)

        clip_model, _ = clip.load("ViT-B/16", device="cpu")
        state = torch.load(ckpt_path, map_location="cpu")
        clip_model.load_state_dict(state["state_dict"], strict=False)

        self.visual = clip_model.visual.float()
        self.embed_dim = embed_dim  # kept for interface parity with FaceEncoder; actual output is always 768-d

        for p in self.visual.parameters():
            p.requires_grad_(False)
        self.visual.eval()

    def train(self, mode=True):
        """Always stays in eval mode - this encoder is never fine-tuned, regardless
        of what TalkingHeadJEPA.train()/.eval() does to the rest of the model."""
        return super().train(False)

    def forward(self, pixels, interpolate_pos_encoding=True):
        """
        Args:
            pixels (Tensor): (Batch*Time, C, H, W), values in [0, 1] - the same
                convention every other representation in this project uses
                (no ImageNet/CLIP mean-std normalization applied).
        Returns:
            A struct with .last_hidden_state of shape (N, 1, 768): FaRL's
            dense patch-token mean, placed at the one and only sequence
            position so jepa.py's existing `output.last_hidden_state[:, 0]`
            pooling works completely unmodified downstream.

        Reproduces CLIP's VisionTransformer.forward (openai/CLIP model.py,
        verified against the live source 2026-08-26, same code path
        validated in JEPA_Latent_Probing.ipynb Part 2g) up to before the
        final CLS-only ln_post + proj step, applying ln_post to every token
        and mean-pooling the patch tokens (excluding CLS) instead of CLIP's
        pooled + projected CLS-only output.
        """
        with torch.no_grad():
            x = pixels.type(self.visual.conv1.weight.dtype)
            x = self.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            cls_tok = self.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([cls_tok, x], dim=1)
            x = x + self.visual.positional_embedding.to(x.dtype)
            x = self.visual.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.visual.ln_post(x)  # applied to ALL tokens, not just x[:, 0]
            patch_mean = x[:, 1:].mean(dim=1).float()  # (N, 768)

        class Output:
            pass

        out = Output()
        out.last_hidden_state = patch_mean.unsqueeze(1)  # (N, 1, 768)
        return out


class TalkingHeadJEPA(nn.Module):
    """
    Talking Head JEPA Prototype.
    
    This wraps the core JEPA module and configures it specifically for processing
    facial video frames and audio features.
    """
    def __init__(
        self,
        image_size=224,
        audio_feature_dim=1024, # e.g., HuBERT large output dim
        embed_dim=256,          # Latent state dimension
        hidden_dim=256,         # Predictor hidden dimension
        history_size=4,         # Context length
        pretrained_imagenet=False,  # Initialize face_encoder from ImageNet weights before JEPA training
        face_encoder_type='vit_b16',  # 'vit_b16' (default, trainable, honors pretrained_imagenet) |
                                       # 'farl_frozen' (added 2026-08-26 - frozen FaRL dense patch
                                       # tokens, no fine-tuning at all; see FarlFrozenFaceEncoder)
    ):
        super().__init__()
        self.history_size = history_size
        self.embed_dim = embed_dim
        self.face_encoder_type = face_encoder_type

        # 1. Face Encoder (Vision)
        if face_encoder_type == 'vit_b16':
            self.face_encoder = FaceEncoder(image_size=image_size, embed_dim=embed_dim,
                                             pretrained_imagenet=pretrained_imagenet)
        elif face_encoder_type == 'farl_frozen':
            self.face_encoder = FarlFrozenFaceEncoder(embed_dim=embed_dim)
        else:
            raise ValueError(f"Unknown face_encoder_type: {face_encoder_type!r} "
                              "(expected 'vit_b16' or 'farl_frozen')")

        # 2. Audio Embedder (Action)
        self.audio_embedder = AudioFeatureEmbedder(
            audio_feature_dim=audio_feature_dim, 
            emb_dim=embed_dim
        )
        
        # 3. Predictor (Autoregressive Transformer)
        self.predictor = ARPredictor(
            num_frames=history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            depth=6,          # Depth for the prototype
            heads=8,
            mlp_dim=hidden_dim * 4,
        )
        
        # 4. Projectors
        # Project vision embeddings to JEPA state space
        self.projector = MLP(
            input_dim=768, # ViT base hidden dim is 768
            output_dim=embed_dim,
            hidden_dim=1024,
            norm_fn=nn.BatchNorm1d,
        )
        
        # Project predictor outputs back to state space
        self.pred_proj = MLP(
            input_dim=embed_dim,
            output_dim=embed_dim,
            hidden_dim=1024,
            norm_fn=nn.BatchNorm1d,
        )
        
        # 5. Core JEPA Model
        self.jepa = JEPA(
            encoder=self.face_encoder,
            predictor=self.predictor,
            action_encoder=self.audio_embedder,
            projector=self.projector,
            pred_proj=self.pred_proj,
        )
        
        # Regularizer
        self.sigreg = SIGReg()
        self.sigreg_weight = 1.0
        
    def encode(self, batch):
        """
        Encode the facial frames and audio features using JEPA.
        """
        return self.jepa.encode(batch)
        
    def forward(self, face_frames, audio_features, num_preds=4):
        """
        Training step forward pass.
        
        Args:
            face_frames (Tensor): Ground truth video frames (B, T, C, H, W)
            audio_features (Tensor): Audio features corresponding to frames (B, T, AudioFeatureDim)
            num_preds (int): Number of future frames to predict
            
        Returns:
            dict: Contains 'loss', 'pred_loss', 'sigreg_loss', 'pred_emb', and 'tgt_emb'
        """
        batch = {
            "pixels": face_frames,
            "action": audio_features # Audio serves as the "action"
        }
        
        # 1. Encode into latent space
        output = self.encode(batch)
        emb = output["emb"]          # (B, T, D)
        act_emb = output["act_emb"]  # (B, T, D)
        
        # 2. Split into context and target
        ctx_len = self.history_size
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        
        tgt_emb = emb[:, num_preds:] # Ground truth future embeddings
        
        # 3. Predict future states
        pred_emb = self.jepa.predict(ctx_emb, ctx_act)
        
        # 4. Compute Losses
        # MSE Prediction Loss
        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        
        # SIG Regularization on the state embeddings to prevent collapse
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        
        # Total Loss
        loss = pred_loss + self.sigreg_weight * sigreg_loss
        
        return {
            "loss": loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "pred_emb": pred_emb,
            "tgt_emb": tgt_emb
        }


# ==========================================
# Testing Block
# ==========================================
if __name__ == "__main__":
    print("Initializing Talking Head JEPA Prototype...")
    model = TalkingHeadJEPA(
        image_size=224,
        audio_feature_dim=1024,
        embed_dim=256,
        hidden_dim=256,
        history_size=4
    )
    
    # Create Dummy Data
    batch_size = 2
    total_time = 8  # 4 context + 4 future
    channels = 3
    height = 224
    width = 224
    audio_dim = 1024
    
    dummy_faces = torch.randn(batch_size, total_time, channels, height, width)
    dummy_audio = torch.randn(batch_size, total_time, audio_dim)
    
    print(f"Input Face Frames Shape: {dummy_faces.shape}")
    print(f"Input Audio Features Shape: {dummy_audio.shape}")
    
    print("Running Forward Pass...")
    outputs = model(dummy_faces, dummy_audio, num_preds=4)
    
    print("Forward Pass Successful!")
    print(f"Total Loss: {outputs['loss'].item():.4f}")
    print(f"Prediction Loss: {outputs['pred_loss'].item():.4f}")
    print(f"SIGReg Loss: {outputs['sigreg_loss'].item():.4f}")
    print(f"Predicted Embeddings Shape: {outputs['pred_emb'].shape}")
    print(f"Target Embeddings Shape: {outputs['tgt_emb'].shape}")
