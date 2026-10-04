import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models.vision_transformer import vit_b_16
# # # # # # # from transformers import HubertModel bypassed bypassed bypassed bypassed bypassed bypassed bypassed

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


class ProjectedAudioExpert(nn.Module):
    """Use Audio Exprt (pre-train model) to encode audio
    """
    def __init__(self, raw_expert_model, raw_output_dim, target_emb_dim):
        super().__init__()
        self.expert = raw_expert_model
        self.proj = nn.Linear(raw_output_dim, target_emb_dim)
        
    def forward(self, x):
        # Depending on the expert, you may need: self.expert(x).last_hidden_state
        features = self.expert(x) 
        return self.proj(features)


# class FaceEncoder(nn.Module):
#     """
#     Vision encoder wrapper for facial frames.
#     Uses a standard PyTorch Vision Transformer (ViT) to encode 
#     224x224 facial frames into latent representations.
#     """
#     def __init__(self, image_size=224, embed_dim=768):
#         super().__init__()
#         # Load a base ViT model without pretrained weights for prototyping
#         self.vit = vit_b_16(weights=None) 
#         self.vit.heads = nn.Identity() # Remove the classification head
#         self.embed_dim = embed_dim
        
#     def forward(self, pixels, interpolate_pos_encoding=True):
#         """
#         Args:
#             pixels (Tensor): Flattened batch of frames (Batch * Time, Channels, Height, Width)
#         Returns:
#             A struct/object with .last_hidden_state containing the sequence with the CLS token.
#             This mimics the HuggingFace ViT output format expected by jepa.py.
#         """
#         # Process input through ViT's patch embedding
#         x = self.vit._process_input(pixels)
#         n = x.shape[0]
        
#         # Append class token
#         batch_class_token = self.vit.class_token.expand(n, -1, -1)
#         x = torch.cat([batch_class_token, x], dim=1)
        
#         # Pass through the transformer encoder
#         x = self.vit.encoder(x)

#         # Wrap output to match the expected format in jepa.py: output.last_hidden_state
#         class Output:
#             pass
#         out = Output()
#         out.last_hidden_state = x  # jepa.py will do x[:, 0]
#         return out

class FaceEncoder(nn.Module):
    def __init__(self, image_size=224, embed_dim=768):
        super().__init__()
        self.vit = vit_b_16(weights=None) 
        self.vit.heads = nn.Identity() 
        self.embed_dim = embed_dim
        
    # FIX: Accept **kwargs so 'interpolate_pos_encoding=True' from jepa.py doesn't crash Python
    def forward(self, pixels, **kwargs):
        # Defensively ensure pixels is 4D (N, C, H, W)
        if len(pixels.shape) == 5:
            pixels = rearrange(pixels, 'b t c h w -> (b t) c h w')
            
        x = self.vit._process_input(pixels)
        n = x.shape[0]
        
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)
        
        class Output:
            pass
        out = Output()
        out.last_hidden_state = x 
        return out


# class TalkingHeadJEPA(nn.Module):
#     """
#     Talking Head JEPA Prototype.
    
#     This wraps the core JEPA module and configures it specifically for processing
#     facial video frames and audio features.
#     """
#     def __init__(
#         self,
#         image_size=224,
#         audio_feature_dim=1024,          # e.g., HuBERT large output dim
#         embed_dim=256,                   # Latent state dimension
#         hidden_dim=256,                  # Predictor hidden dimension
#         history_size=4,                  # Context length
#         audio_expert=None,               # Use pre-train model to encode audio
#         freeze_audio_expert_params=False # If freeze param of audio or not
#     ):
#         super().__init__()
#         self.history_size = history_size
#         self.embed_dim = embed_dim
        
#         # 1. Face Encoder (Vision)
#         self.face_encoder = FaceEncoder(image_size=image_size, embed_dim=embed_dim)

        
#         # 2. Audio Embedder (Action)
#         if not audio_expert:
#             self.audio_embedder = AudioFeatureEmbedder(
#                 audio_feature_dim=audio_feature_dim, 
#                 emb_dim=embed_dim
#             )
#         else:
#             self.audio_embedder = audio_expert
#             # Freeze all parameters
#             if freeze_audio_expert_params:
#                 for param in self.audio_embedder.parameters():
#                     param.requires_grad = False
        
#         # 3. Predictor (Autoregressive Transformer)
#         self.predictor = ARPredictor(
#             num_frames=history_size,
#             input_dim=embed_dim,
#             hidden_dim=hidden_dim,
#             output_dim=embed_dim,
#             depth=6,          # Depth for the prototype
#             heads=8,
#             mlp_dim=hidden_dim * 4,
#         )
        
#         # 4. Projectors
#         # Project vision embeddings to JEPA state space
#         self.projector = MLP(
#             input_dim=768, # ViT base hidden dim is 768
#             output_dim=embed_dim,
#             hidden_dim=1024,
#             norm_fn=nn.BatchNorm1d,
#         )
        
#         # Project predictor outputs back to state space
#         self.pred_proj = MLP(
#             input_dim=embed_dim,
#             output_dim=embed_dim,
#             hidden_dim=1024,
#             norm_fn=nn.BatchNorm1d,
#         )
        
#         # 5. Core JEPA Model
#         self.jepa = JEPA(
#             encoder=self.face_encoder,
#             predictor=self.predictor,
#             action_encoder=self.audio_embedder,
#             projector=self.projector,
#             pred_proj=self.pred_proj,
#         )
        
#         # Regularizer
#         self.sigreg = SIGReg()
#         self.sigreg_weight = 1.0
        
#     def encode(self, batch):
#         """
#         Encode the facial frames and audio features using JEPA.
#         """
#         return self.jepa.encode(batch)
        
#     def forward(self, face_frames, audio_features, num_preds=4):
#         """
#         Training step forward pass.
        
#         Args:
#             face_frames (Tensor): Ground truth video frames (B, T, C, H, W)
#             audio_features (Tensor): Audio features corresponding to frames (B, T, AudioFeatureDim)
#             num_preds (int): Number of future frames to predict
            
#         Returns:
#             dict: Contains 'loss', 'pred_loss', 'sigreg_loss', 'pred_emb', and 'tgt_emb'
#         """
#         print("RAW frames shape from dataloader:", face_frames.shape)
#         print("RAW audio shape from dataloader:", audio_features.shape)

#         B, T, C, H, W = face_frames.shape
#         # Flatten Time dimension for Vision Transformer processing
#         flat_frames = rearrange(face_frames, 'b t c h w -> (b t) c h w')

#         batch = {
#             "pixels": flat_frames, # Use flat_frames instead of use face_frames directly
#             "action": audio_features # Audio serves as the "action"
#         }
        
#         # 1. Encode into latent space
#         output = self.encode(batch)
#         # Unflatten state embeddings back to temporal layout (B, T, D)
#         act_emb = output["act_emb"]  # Action representations (B, T, D)
#         raw_emb = output["emb"]
#         if len(raw_emb.shape) == 3: # If it's (B*T, Tokens, Dimension)
#             raw_emb = raw_emb[:, 0] # Extract CLS token -> (B*T, Dimension)

#         # Restore temporal structure: (B*T, D) -> (B, T, D)
#         emb = rearrange(raw_emb, '(b t) d -> b t d', b=B, t=T)

#         # 2. Split into context and target
#         ctx_len = self.history_size
#         ctx_emb = emb[:, :ctx_len]

#         # If predicting 4 frames into the future, we need the actions guiding those 4 steps
#         ctx_act = act_emb[:, :ctx_len + num_preds]
        
#         # Slice Target accurately to match exact prediction length
#         # Target Embeddings should be detached from current graph updates
#         tgt_emb = emb[:, ctx_len : ctx_len + num_preds].detach() # Ground truth future embeddings
        
#         # 3. Predict future states
#



        if ctx_act.shape[-1] != 768:
            if not hasattr(self, 'safety_audio_proj'):
                self.safety_audio_proj = nn.Linear(ctx_act.shape[-1], 768).to(ctx_act.device)
            ctx_act = self.safety_audio_proj(ctx_act)
        pred_emb = self.jepa.predict(ctx_emb, ctx_act)

        
#         # 4. Compute Losses
#         # MSE Prediction Loss
#         pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        
#         # SIG Regularization on the state embeddings to prevent collapse
#         sigreg_loss = self.sigreg(emb.transpose(0, 1))
        
#         # Total Loss
#         loss = pred_loss + self.sigreg_weight * sigreg_loss
        
#         return {
#             "loss": loss,
#             "pred_loss": pred_loss,
#             "sigreg_loss": sigreg_loss,
#             "pred_emb": pred_emb,
#             "tgt_emb": tgt_emb
#         }

class TalkingHeadJEPA(nn.Module):
    """
    Talking Head JEPA Prototype (Patched Dimension-Bypass Version).
    """
    def __init__(
        self,
        image_size=224,
        audio_feature_dim=1024,          
        embed_dim=256,                   
        hidden_dim=256,                  
        history_size=4,                  
        audio_expert=None,               
        freeze_audio_expert_params=False 
    ):
        super().__init__()
        self.history_size = history_size
        self.embed_dim = embed_dim
        
        # 1. Face Encoder (Vision)
        self.face_encoder = FaceEncoder(image_size=image_size, embed_dim=embed_dim)
        
        # 2. Audio Embedder (Action)
        if not audio_expert:
            self.audio_embedder = AudioFeatureEmbedder(
                audio_feature_dim=audio_feature_dim, 
                emb_dim=embed_dim
            )
        else:
            self.audio_embedder = audio_expert
            if freeze_audio_expert_params:
                for param in self.audio_embedder.parameters():
                    param.requires_grad = False
        
        # 3. Predictor (Autoregressive Transformer)
        self.predictor = ARPredictor(
            num_frames=history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            depth=6,          
            heads=8,
            mlp_dim=hidden_dim * 4,
        )
        
        # 4. Projectors
        self.projector = MLP(input_dim=768, output_dim=embed_dim, hidden_dim=1024, norm_fn=nn.BatchNorm1d)
        self.pred_proj = MLP(input_dim=embed_dim, output_dim=embed_dim, hidden_dim=1024, norm_fn=nn.BatchNorm1d)
        
        # 5. Core JEPA Model
        self.jepa = JEPA(
            encoder=self.face_encoder,
            predictor=self.predictor,
            action_encoder=self.audio_embedder,
            projector=self.projector,
            pred_proj=self.pred_proj,
        )
        
        self.sigreg = SIGReg()
        self.sigreg_weight = 1.0
        
    def encode(self, batch):
        """
        Bypasses jepa.py's internal masking to prevent 3D channel destruction.
        Manually processes the custom structured batch.
        """
        # A. Process Frames directly using your FaceEncoder and Projector
        # This completely guarantees jepa.py cannot touch or break your 4D shapes!
        encoder_out = self.face_encoder(batch["pixels"], interpolate_pos_encoding=True)
        raw_features = encoder_out.last_hidden_state # Expected shape: (B*T, Tokens, 768) or (B*T, 768)
        
        # Handle sequence token structures smoothly
        if len(raw_features.shape) == 3:
            raw_features = raw_features[:, 0] # Extract the CLS token
            
        emb = self.projector(raw_features) # Map 768 down to your embed_dim (256)
        
        # B. Process Audio actions cleanly
        act_emb = self.audio_embedder(batch["action"]) # Maps (B, T, AudioDim) -> (B, T, 256)
        
        return {
            "emb": emb,       # Output matches the flat layout: (B*T, 256)
            "act_emb": act_emb # Output matches standard layout: (B, T, 256)
        }
        
    def forward(self, face_frames, audio_features, num_preds=4):
        B, T, C, H, W = face_frames.shape
        
        # Flatten Time dimension cleanly for standard image processing layouts
        flat_frames = rearrange(face_frames, 'b t c h w -> (b t) c h w')

        batch = {
            "pixels": flat_frames, 
            "action": audio_features 
        }
        
        # 1. Safely run our customized, bypass-protected encoder method
        output = self.encode(batch)
        act_emb = output["act_emb"]  # (B, T, D)
        
        # Restore spatial-temporal groupings: (B*T, D) -> (B, T, D)
        emb = rearrange(output["emb"], '(b t) d -> b t d', b=B, t=T)

        # 2. Split into context and target horizons
        ctx_len = self.history_size
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len + num_preds]
        
        # Isolate target updates to ensure stability
        tgt_emb = emb[:, ctx_len : ctx_len + num_preds].detach() 
        
        # 3. Predict future states using the JEPA infrastructure



        if ctx_act.shape[-1] != 768:
            if not hasattr(self, 'safety_audio_proj'):
                self.safety_audio_proj = nn.Linear(ctx_act.shape[-1], 768).to(ctx_act.device)
            ctx_act = self.safety_audio_proj(ctx_act)
        pred_emb = self.jepa.predict(ctx_emb, ctx_act)

        
        # 4. Compute Losses
        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
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