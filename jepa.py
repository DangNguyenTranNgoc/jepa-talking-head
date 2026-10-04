"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):
    """
    Joint-Embedding Predictive Architecture (JEPA).
    
    This is the core World Model architecture. It encodes observations (e.g., video frames)
    and actions (e.g., audio features for talking head synthesis) into embeddings,
    and autoregressively predicts future state embeddings based on the action/audio conditions.
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        """
        Args:
            encoder (nn.Module): Vision encoder (e.g., ViT) to process pixel inputs.
            predictor (nn.Module): Autoregressive transformer predicting future states.
            action_encoder (nn.Module): Embedder for the action space (or audio features).
            projector (nn.Module, optional): Projects vision encoder features to state embedding space.
            pred_proj (nn.Module, optional): Projects predictor output to state embedding space.
        """
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        
        Args:
            info (dict): Dictionary containing 'pixels' (video frames) and optionally 'action' 
                         (which could be audio features for talking head generation).
                         - 'pixels' shape: (Batch, Time, Channels, Height, Width)
                         
        Returns:
            dict: Updated info dictionary containing 'emb' (state embeddings) and 'act_emb' (action/audio embeddings).
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        # Flatten batch and time dimensions for the vision encoder
        pixels = rearrange(pixels, "b t ... -> (b t) ...") 
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        # Extract the CLS token as the representation of the frame
        pixels_emb = output.last_hidden_state[:, 0]  
        # Project to the latent state dimension
        emb = self.projector(pixels_emb)
        # Unflatten back to (Batch, Time, Embedding_Dim)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            # Encode the action or audio condition into an embedding space
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Autoregressively predict the next state embedding given context and action/audio.
        
        Args:
            emb (Tensor): Context state embeddings. Shape: (Batch, Time, Dim)
            act_emb (Tensor): Action or Audio embeddings. Shape: (Batch, Time, Action_Dim)
            
        Returns:
            Tensor: Predicted future state embeddings. Shape: (Batch, Time, Dim)
        """
        preds = self.predictor(emb, act_emb)
        # Project the prediction back into the state embedding space
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        
        This is used during inference to autoregressively unroll future states based on
        a sequence of future actions (or future audio features for talking head synthesis).

        Args:
            info (dict): Initial observation containing 'pixels'. Shape: (Batch, Samples, TimeContext, C, H, W)
            action_sequence (Tensor): Planned future actions (or audio features). Shape: (Batch, Samples, TotalTime, ActionDim)
                - Samples (S) is the number of action plan samples being evaluated.
                - TotalTime (T) is the total time horizon.
            history_size (int): Number of context frames to feed into the autoregressive predictor.
            
        Returns:
            dict: Updated info dict with 'predicted_emb' containing the unrolled future state embeddings.
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings.
        
        This calculates the Mean Squared Error (MSE) loss in the embedding space 
        between what the model predicted and the target representations.
        
        Args:
            info_dict (dict): Dictionary containing 'predicted_emb' and 'goal_emb'.
            
        Returns:
            Tensor: Cost per action candidate. Shape: (Batch, Samples)
        """
        pred_emb = info_dict["predicted_emb"]  # Predicted states (B, S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # Ground truth goal states (B, S, T, dim)

        # Align the temporal dimension for comparison
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # Compute Mean Squared Error across embedding dimensions, return last-step cost
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state.
        
        This unrolls the state into the future for multiple action candidates and 
        scores them based on how close the predicted states get to the goal state.
        """

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
