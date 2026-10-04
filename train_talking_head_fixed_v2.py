import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from transformers import HubertModel

from talking_head_jepa_fixed_v2 import TalkingHeadJEPA

class FaceDecoder(nn.Module):
    """
    A simple transposed convolutional decoder to reconstruct 224x224 RGB faces
    from the 256-dimensional latent embeddings predicted by JEPA.
    This is required to export predicted videos.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 256 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),    # 224x224
            nn.Sigmoid() # Output pixels in [0, 1]
        )

    def forward(self, x):
        # x: (Batch, Time, embed_dim)
        b, t, d = x.shape
        x = x.reshape(b * t, d)
        x = self.fc(x)
        x = x.reshape(b * t, 256, 7, 7)
        x = self.deconv(x)
        x = x.reshape(b, t, 3, 224, 224)
        return x

class DummySyncTalkDataset(Dataset):
    """
    Dummy Dataset mimicking the SyncTalk data loader.
    Yields sequences of facial frames and corresponding audio features.
    """
    def __init__(self, num_samples=100, seq_len=8):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return random tensors mimicking shape (Time, Channels, Height, Width)
        # and audio features (Time, AudioFeatureDim)
        frames = torch.rand(self.seq_len, 3, 224, 224)
        audio = torch.randn(self.seq_len, 1024)
        return frames, audio

@hydra.main(version_base=None, config_path="./config/train", config_name="talking_head")
def train(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use Audio Expert model
    audio_expert = torch.nn.Identity()

    # Initialize models
    model = TalkingHeadJEPA(
        image_size=cfg.model.image_size,
        audio_feature_dim=cfg.model.audio_feature_dim,
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        history_size=cfg.model.history_size,
        audio_expert=audio_expert,
        freeze_audio_expert_params=True
    ).to(device)
    
    decoder = FaceDecoder(embed_dim=cfg.model.embed_dim).to(device)

    # Dataset & Dataloader
    dataset = DummySyncTalkDataset(num_samples=cfg.num_samples, seq_len=cfg.seq_len)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(decoder.parameters()), lr=cfg.lr)

    # Training loop
    epochs = cfg.epochs
    model.train()
    decoder.train()
    
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (frames, audio) in enumerate(progress_bar):
            frames, audio = frames.to(device), audio.to(device)

            optimizer.zero_grad()

            # 1. Forward pass through JEPA
            outputs = model(frames, audio, num_preds=cfg.model.num_preds)
            jepa_loss = outputs["loss"]
            
            # 2. Decode the target embeddings to compute reconstruction loss (Optional but necessary to train decoder)
            tgt_frames = frames[:, cfg.model.history_size : cfg.model.history_size + cfg.model.num_preds] # Ground truth future frames
            decoded_frames = decoder(outputs["tgt_emb"]) # Reconstruct from true target embeddings to train decoder
            
            # MSE loss for reconstruction
            recon_loss = nn.functional.mse_loss(decoded_frames, tgt_frames)
            
            # Total loss
            total_loss = jepa_loss + recon_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            progress_bar.set_postfix({"Loss": total_loss.item(), "JEPA": jepa_loss.item(), "Recon": recon_loss.item()})
            
        print(f"Epoch {epoch+1} average loss: {epoch_loss / len(dataloader):.4f}")
        
        # Save checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"checkpoints/talking_head_ckpt_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
