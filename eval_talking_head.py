import os
import torch
import torchvision
import torchvision.io
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from talking_head_jepa import TalkingHeadJEPA
from train_talking_head import FaceDecoder, DummySyncTalkDataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="talking_head")
def evaluate_and_export(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models
    model = TalkingHeadJEPA(
        image_size=cfg.model.image_size,
        audio_feature_dim=cfg.model.audio_feature_dim,
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        history_size=cfg.model.history_size
    ).to(device)
    
    decoder = FaceDecoder(embed_dim=cfg.model.embed_dim).to(device)

    # Load checkpoint if available
    ckpt_path = cfg.checkpoint_path
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    else:
        print("No checkpoint found! Using untrained prototype weights for evaluation.")

    model.eval()
    decoder.eval()

    # Test Dataset
    dataset = DummySyncTalkDataset(num_samples=cfg.num_samples, seq_len=cfg.seq_len)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    os.makedirs("results", exist_ok=True)

    print("Evaluating and exporting predicted videos...")
    with torch.no_grad():
        for batch_idx, (frames, audio) in enumerate(dataloader):
            frames, audio = frames.to(device), audio.to(device)

            # 1. Run JEPA to predict future latent states
            outputs = model(frames, audio, num_preds=cfg.model.num_preds)
            pred_emb = outputs["pred_emb"] # Shape: (B, num_preds, embed_dim)
            
            # Compute Latent Evaluation Metric (MSE in embedding space)
            tgt_emb = outputs["tgt_emb"]
            latent_mse = torch.nn.functional.mse_loss(pred_emb, tgt_emb)
            print(f"Batch {batch_idx} - Latent MSE: {latent_mse.item():.4f}")

            # 2. Decode the predicted latent states to generate the output video
            predicted_frames = decoder(pred_emb) # Shape: (1, 4, 3, 224, 224)
            
            # 3. Export to video
            video_tensor = predicted_frames[0].permute(0, 2, 3, 1) # (4, 224, 224, 3)
            video_numpy = (video_tensor * 255).byte().cpu().numpy()
            
            out_path = f"results/predicted_video_batch_{batch_idx}.mp4"
            import imageio
            imageio.mimwrite(out_path, video_numpy, fps=25, macro_block_size=None)
            print(f"Exported predicted video to: {out_path}")

if __name__ == "__main__":
    evaluate_and_export()
