import torch
import torch.nn.functional as F # For F.one_hot
# import os # Not used in this specific function directly
# import matplotlib.pyplot as plt # Not used in this specific function directly
# import numpy as np # Not used in this specific function directly
# import time # Not used in this specific function directly
import wandb # For wandb.Image

"""Handles the training loop for the reward predictor model."""

def train_reward_mlp_epoch(
    reward_mlp_model, base_model, optimizer_reward_mlp, train_dataloader,
    loss_fn, device, action_dim, action_type,
    model_name_log_prefix, num_epochs_reward_mlp, log_interval_reward_mlp,
    is_jepa_base_model, # Boolean to differentiate input processing
    wandb_run
):
    """
    Handles the training process for a reward MLP (Multi-Layer Perceptron) model
    over a specified number of epochs.

    This function trains an MLP to predict rewards based on embeddings/features
    extracted from a base model (either a Standard Encoder-Decoder or a JEPA model).
    It iterates through the training data, computes losses, performs backpropagation,
    and updates the optimizer. Training progress (loss, learning rate) is logged
    using Weights & Biases (wandb).

    Args:
        reward_mlp_model (torch.nn.Module): The reward MLP model to be trained.
        base_model (torch.nn.Module): The base model (StdEncDec or JEPA) used for
                                      feature extraction. It's set to eval mode.
        optimizer_reward_mlp (torch.optim.Optimizer): Optimizer for the reward MLP.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training set.
        loss_fn (callable): The loss function for training the reward MLP.
        device (torch.device): The device (CPU or GPU) for computations.
        action_dim (int): Dimension of the action space.
        action_type (str): Type of action space ('discrete' or 'continuous').
        model_name_log_prefix (str): Prefix for logging (e.g., "Reward MLP (Enc-Dec)").
        num_epochs_reward_mlp (int): Number of epochs to train the reward MLP.
        log_interval_reward_mlp (int): Interval (in batches) for logging training.
        is_jepa_base_model (bool): Flag indicating if the base_model is a JEPA model.
                                   This affects how input features are derived.
        wandb_run (wandb.sdk.wandb_run.Run, optional): Active Weights & Biases run object.

    Note:
        This function doesn't return any value but modifies the `reward_mlp_model`
        in place and logs metrics to wandb.
    """
    if not (reward_mlp_model and base_model and optimizer_reward_mlp and train_dataloader):
        print(f"{model_name_log_prefix}: Components missing, skipping training.")
        return

    # Determine wandb prefix based on the base model type
    wandb_model_prefix = "JEPA" if is_jepa_base_model else "StdEncDec"

    # move model to device
    reward_mlp_model.to(device)

    for epoch in range(num_epochs_reward_mlp):
        reward_mlp_model.train()
        if base_model: base_model.eval() # Base model is used for feature extraction

        epoch_loss_reward_mlp = 0
        num_train_batches = len(train_dataloader)

        if num_train_batches == 0:
            print(f"{model_name_log_prefix} Epoch {epoch+1}: No training data. Skipping.")
            continue

        for batch_idx, (s_t, a_t, r_t, s_t_plus_1) in enumerate(train_dataloader):
            s_t, r_t, s_t_plus_1 = s_t.to(device), r_t.to(device).float().unsqueeze(1), s_t_plus_1.to(device)

            if action_type == 'discrete':
                if a_t.ndim == 1: a_t = a_t.unsqueeze(1)
                a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
            else:
                a_t_processed = a_t.float().to(device)

            optimizer_reward_mlp.zero_grad()

            input_to_reward_mlp = None
            with torch.no_grad():
                if is_jepa_base_model:
                    # For JEPA, use s_t, a_t, s_t_plus_1 to get predictor embedding
                    pred_emb_for_reward, _, _, _ = base_model(s_t, a_t_processed, s_t_plus_1)
                    input_to_reward_mlp = pred_emb_for_reward.detach()
                else: # For StdEncDec
                    predicted_s_t_plus_1_for_reward = base_model(s_t, a_t_processed)
                    input_to_reward_mlp = predicted_s_t_plus_1_for_reward.view(predicted_s_t_plus_1_for_reward.size(0), -1).detach()

            if input_to_reward_mlp is None:
                print(f"{model_name_log_prefix} Epoch {epoch+1}, Batch {batch_idx+1}: Failed to get input from base model. Skipping batch.")
                continue

            pred_reward = reward_mlp_model(input_to_reward_mlp)
            loss_reward = loss_fn(pred_reward, r_t)

            loss_reward.backward()
            optimizer_reward_mlp.step()

            loss_reward_item = loss_reward.item()
            epoch_loss_reward_mlp += loss_reward_item

            if (batch_idx + 1) % log_interval_reward_mlp == 0:
                if wandb_run:
                    log_data_reward_batch = {
                        f"reward_mlp/{wandb_model_prefix}/train/Loss": loss_reward_item,
                        f"reward_mlp/{wandb_model_prefix}/train/Learning_Rate": optimizer_reward_mlp.param_groups[0]['lr']
                    }
                    # The step current_reward_mlp_global_step should align with f"{wandb_model_prefix}/reward_mlp/train/step"
                    wandb_run.log(log_data_reward_batch)

        avg_epoch_loss_reward_mlp = epoch_loss_reward_mlp / num_train_batches if num_train_batches > 0 else 0
        print(f"  Epoch {epoch+1}/{num_epochs_reward_mlp} Avg Train Loss: {avg_epoch_loss_reward_mlp:.4f} ")
        if wandb_run:
            # Log using epoch + 1 to align with f"{wandb_model_prefix}/reward_mlp/epoch"
            wandb_run.log({
                f"reward_mlp/{wandb_model_prefix}/train_epoch_avg/Loss": avg_epoch_loss_reward_mlp
            })

    print(f"{model_name_log_prefix} training finished.")
    # Optionally return last epoch's average loss or a status
    # For now, no explicit return value is critical for the flow.
