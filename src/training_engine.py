# Contents for src/training_engine.py
import torch
import torch.nn.functional as F # For F.one_hot
import os # For os.path.exists in early stopping save/load
import matplotlib.pyplot as plt
import numpy as np

# Note: Loss functions (mse_loss_fn, aux_loss_fn, aux_loss_name, aux_loss_weight)
# will be passed in via the 'losses_map' dictionary.
# Models (std_enc_dec, jepa_model, reward_mlp_enc_dec, reward_mlp_jepa, jepa_decoder) via 'models_map'.
# Optimizers (optimizer_std_enc_dec, etc.) via 'optimizers_map'.
# Dataloaders (train_dataloader, val_dataloader) via 'dataloaders_map'.
# Configs (early_stopping_config, enc_dec_mlp_config, jepa_mlp_config, main_config for num_epochs, log_interval) via 'config'.
# Device, action_dim, action_type also passed as arguments.

def _train_validate_model_epoch(
    model, optimizer, train_dataloader, val_dataloader,
    loss_fn, aux_loss_fn, aux_loss_name, aux_loss_weight,
    device, epoch_num, log_interval, action_dim, action_type,
    early_stopping_state, checkpoint_path, model_name_log_prefix,
    update_target_fn=None # For JEPA's target network update
):
    """
    Handles training and validation for one epoch for a given model.
    Returns updated early_stopping_state and epoch losses.
    """
    # === Training Phase ===
    model.train()
    if aux_loss_fn and hasattr(aux_loss_fn, 'train'):
        aux_loss_fn.train()

    epoch_train_loss_primary = 0
    epoch_train_loss_aux = 0
    num_train_batches = len(train_dataloader) if train_dataloader else 0

    if num_train_batches == 0:
        print(f"{model_name_log_prefix} Epoch {epoch_num}: No training data. Skipping training phase.")
        # Return current state if no training data
        return early_stopping_state, 0, 0 # primary_loss, aux_loss

    for batch_idx, (s_t, a_t, r_t, s_t_plus_1) in enumerate(train_dataloader):
        s_t, s_t_plus_1 = s_t.to(device), s_t_plus_1.to(device)
        # r_t is not used by std_enc_dec or jepa directly, but is part of dataloader structure

        if action_type == 'discrete':
            if a_t.ndim == 1: a_t = a_t.unsqueeze(1)
            a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
        else:
            a_t_processed = a_t.float().to(device)

        if batch_idx == 0 and epoch_num == 1:
            print(f"Debug Devices ({model_name_log_prefix} - Train/Val Epoch):")
            print(f"  s_t device: {s_t.device}")
            print(f"  a_t_processed device: {a_t_processed.device}")
            print(f"  model device (first param): {next(model.parameters()).device}")

        optimizer.zero_grad()

        current_loss_primary_item = 0
        current_loss_aux_item = 0
        total_loss_item = 0

        if model_name_log_prefix == "JEPA": # Specific logic for JEPA
            pred_emb, target_emb_detached, online_s_t_emb, online_s_t_plus_1_emb = model(s_t, a_t_processed, s_t_plus_1)
            loss_primary = loss_fn(pred_emb, target_emb_detached)
            current_loss_primary_item = loss_primary.item()

            current_loss_aux = torch.tensor(0.0, device=device)
            if aux_loss_fn is not None and aux_loss_weight > 0:
                aux_term_s_t, _, _ = aux_loss_fn.calculate_reg_terms(online_s_t_emb)
                if online_s_t_plus_1_emb is not None:
                    aux_term_s_t_plus_1, _, _ = aux_loss_fn.calculate_reg_terms(online_s_t_plus_1_emb)
                    current_loss_aux = (aux_term_s_t + aux_term_s_t_plus_1) * 0.5
                else: # online_s_t_plus_1_emb is None (e.g., for vjepa2 mode)
                    current_loss_aux = aux_term_s_t
            current_loss_aux_item = current_loss_aux.item()
            total_loss = loss_primary + current_loss_aux * aux_loss_weight
            total_loss_item = total_loss.item()
        else: # Standard Encoder-Decoder or other models
            predicted_s_t_plus_1 = model(s_t, a_t_processed)
            loss_primary = loss_fn(predicted_s_t_plus_1, s_t_plus_1)
            current_loss_primary_item = loss_primary.item()
            # No separate aux loss calculation here for std_enc_dec in this structure
            total_loss = loss_primary
            total_loss_item = current_loss_primary_item


        total_loss.backward()
        optimizer.step()

        if update_target_fn: # For JEPA
            update_target_fn()

        epoch_train_loss_primary += current_loss_primary_item
        epoch_train_loss_aux += current_loss_aux_item # Will be 0 if not JEPA or no aux loss

        if (batch_idx + 1) % log_interval == 0:
            log_msg = f"  {model_name_log_prefix} Epoch {epoch_num}, Batch {batch_idx+1}/{num_train_batches}:"
            if model_name_log_prefix == "JEPA":
                weighted_aux_loss_str = f"{(current_loss_aux_item * aux_loss_weight):.4f}" if aux_loss_fn and aux_loss_weight > 0 else "N/A"
                log_msg += (f" Pred L: {current_loss_primary_item:.4f},"
                            f" {aux_loss_name} AuxRawL: {current_loss_aux_item:.4f} (W: {weighted_aux_loss_str}),"
                            f" Total L: {total_loss_item:.4f}")
            else: # StdEncDec
                log_msg += f" Loss: {current_loss_primary_item:.4f}"
            print(log_msg)

    avg_epoch_train_loss_primary = epoch_train_loss_primary / num_train_batches
    avg_epoch_train_loss_aux = epoch_train_loss_aux / num_train_batches if aux_loss_fn and aux_loss_weight > 0 else 0


    # === Validation Phase ===
    if val_dataloader:
        model.eval()
        if aux_loss_fn and hasattr(aux_loss_fn, 'eval'):
            aux_loss_fn.eval()

        epoch_val_loss_primary = 0
        epoch_val_loss_aux = 0
        num_val_batches = len(val_dataloader)

        with torch.no_grad():
            for s_t_val, a_t_val, r_t_val, s_t_plus_1_val in val_dataloader:
                s_t_val, s_t_plus_1_val = s_t_val.to(device), s_t_plus_1_val.to(device)
                if action_type == 'discrete':
                    if a_t_val.ndim == 1: a_t_val = a_t_val.unsqueeze(1)
                    a_t_val_processed = F.one_hot(a_t_val.long().view(-1), num_classes=action_dim).float().to(device)
                else:
                    a_t_val_processed = a_t_val.float().to(device)

                val_loss_primary_item = 0
                val_loss_aux_item = 0

                if model_name_log_prefix == "JEPA":
                    pred_emb_val, target_emb_detached_val, online_s_t_emb_val, online_s_t_plus_1_emb_val = model(s_t_val, a_t_val_processed, s_t_plus_1_val)
                    val_loss_primary = loss_fn(pred_emb_val, target_emb_detached_val)
                    val_loss_primary_item = val_loss_primary.item()

                    current_val_loss_aux = torch.tensor(0.0, device=device)
                    if aux_loss_fn is not None and aux_loss_weight > 0:
                        aux_term_s_t_val, _, _ = aux_loss_fn.calculate_reg_terms(online_s_t_emb_val)
                        if online_s_t_plus_1_emb_val is not None:
                            aux_term_s_t_plus_1_val, _, _ = aux_loss_fn.calculate_reg_terms(online_s_t_plus_1_emb_val)
                            current_val_loss_aux = (aux_term_s_t_val + aux_term_s_t_plus_1_val) * 0.5
                        else: # online_s_t_plus_1_emb_val is None
                            current_val_loss_aux = aux_term_s_t_val
                    val_loss_aux_item = current_val_loss_aux.item()
                else: # Standard Encoder-Decoder
                    predicted_s_t_plus_1_val = model(s_t_val, a_t_val_processed)
                    val_loss_primary = loss_fn(predicted_s_t_plus_1_val, s_t_plus_1_val)
                    val_loss_primary_item = val_loss_primary.item()
                    # No separate aux loss for std_enc_dec

                epoch_val_loss_primary += val_loss_primary_item
                epoch_val_loss_aux += val_loss_aux_item

        avg_val_loss_primary = epoch_val_loss_primary / num_val_batches if num_val_batches > 0 else float('inf')
        avg_val_loss_aux_raw = epoch_val_loss_aux / num_val_batches if num_val_batches > 0 and aux_loss_fn and aux_loss_weight > 0 else 0

        # Total validation loss for early stopping decision
        current_total_val_loss = avg_val_loss_primary
        if model_name_log_prefix == "JEPA" and aux_loss_fn and aux_loss_weight > 0:
            current_total_val_loss += avg_val_loss_aux_raw * aux_loss_weight

        print(f"--- {model_name_log_prefix} Epoch {epoch_num} Validation Summary ---")
        if model_name_log_prefix == "JEPA":
            print(f"  Avg Val Pred L: {avg_val_loss_primary:.4f}, {aux_loss_name} AuxRawL: {avg_val_loss_aux_raw:.4f}, Total Val L: {current_total_val_loss:.4f}")
        else:
            print(f"  Avg Val Loss: {avg_val_loss_primary:.4f}")


        # Early Stopping Logic
        if current_total_val_loss < early_stopping_state['best_val_loss'] - early_stopping_state.get('delta', 0.001):
            early_stopping_state['best_val_loss'] = current_total_val_loss
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
            early_stopping_state['epochs_no_improve'] = 0
            print(f"  {model_name_log_prefix}: Val loss improved. Saved model to {checkpoint_path}")
        else:
            early_stopping_state['epochs_no_improve'] += 1
            print(f"  {model_name_log_prefix}: No val improvement for {early_stopping_state['epochs_no_improve']} epochs.")
            if early_stopping_state['epochs_no_improve'] >= early_stopping_state.get('patience', 10):
                early_stopping_state['early_stop_flag'] = True
                print(f"  {model_name_log_prefix}: Early stopping triggered.")
    # else: No validation dataloader
        # Original behavior for main models (StdEncDec, JEPA) was to not save if no validation.
        # Checkpointing was tied to validation loss improvement.
        # This 'else' block is intentionally left without a save for main models.
        # The JEPA State Decoder has its own specific handling for no-validation saving.


    return early_stopping_state, avg_epoch_train_loss_primary, avg_epoch_train_loss_aux


def _train_reward_mlp_epoch(
    reward_mlp_model, base_model, optimizer_reward_mlp, train_dataloader,
    loss_fn, device, action_dim, action_type,
    model_name_log_prefix, num_epochs_reward_mlp, log_interval_reward_mlp,
    is_jepa_base_model # Boolean to differentiate input processing
):
    """
    Handles training for a reward MLP model for a specified number of epochs.
    """
    if not (reward_mlp_model and base_model and optimizer_reward_mlp and train_dataloader):
        print(f"{model_name_log_prefix}: Components missing, skipping training.")
        return

    print(f"\nStarting training for {model_name_log_prefix} for {num_epochs_reward_mlp} epochs...")

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

            if batch_idx == 0 and epoch == 0: # epoch here is 0-indexed
                print(f"Debug Devices ({model_name_log_prefix} - Reward MLP Epoch):")
                print(f"  s_t device: {s_t.device}")
                print(f"  a_t_processed device: {a_t_processed.device}")
                # For r_t and s_t_plus_1, they are already on device from above
                print(f"  r_t device: {r_t.device}")
                print(f"  s_t_plus_1 device: {s_t_plus_1.device}")
                print(f"  reward_mlp_model device (first param): {next(reward_mlp_model.parameters()).device}")
                if base_model:
                    print(f"  base_model device (first param): {next(base_model.parameters()).device}")


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
                print(f"  {model_name_log_prefix} Epoch {epoch+1}, Batch {batch_idx+1}/{num_train_batches}: Loss {loss_reward_item:.4f}")

        avg_epoch_loss_reward_mlp = epoch_loss_reward_mlp / num_train_batches if num_train_batches > 0 else 0
        print(f"--- {model_name_log_prefix} Epoch {epoch+1}/{num_epochs_reward_mlp} Summary: Avg Train Loss {avg_epoch_loss_reward_mlp:.4f} ---")

    print(f"{model_name_log_prefix} training finished.")
    # Optionally return last epoch's average loss or a status
    # For now, no explicit return value is critical for the flow.

def _train_jepa_state_decoder(
    jepa_decoder_model, jepa_model, optimizer_jepa_decoder,
    train_dataloader, val_dataloader, loss_fn, device,
    action_dim, action_type,
    decoder_training_config, # This is jepa_decoder_training_config from the main function
    main_model_dir, # This is model_dir from the main function, for paths
    general_log_interval # Fallback log interval
):
    """
    Handles training for the JEPA State Decoder model.
    Returns the path to the best saved checkpoint, or None.
    """
    print("\nStarting JEPA State Decoder training...")

    # Extract params from decoder_training_config
    num_epochs_decoder = decoder_training_config.get('num_epochs', 50)
    decoder_log_interval = decoder_training_config.get('log_interval', general_log_interval)

    early_stopping_specific_config = decoder_training_config.get('early_stopping', {})
    # Fallback to general patience/delta if not specifically in decoder_training_config.early_stopping
    # This requires access to the general patience/delta, or they should be passed in,
    # or defined within this scope if they are meant to be independent.
    # For now, let's assume patience/delta are in early_stopping_specific_config or have defaults.
    patience_decoder = early_stopping_specific_config.get('patience', 10) # Default if not in specific config
    delta_decoder = early_stopping_specific_config.get('delta', 0.001)   # Default if not in specific config

    decoder_cp_name = decoder_training_config.get('checkpoint_path', 'best_jepa_decoder.pth')
    # Construct full checkpoint path using main_model_dir
    checkpoint_path_decoder = os.path.join(main_model_dir, decoder_cp_name)

    # Plotting directory from config
    validation_plot_dir_config = decoder_training_config.get('validation_plot_dir', "validation_plots/decoder")
    validation_plot_dir_full = os.path.join(main_model_dir, validation_plot_dir_config) # Construct full path

    early_stopping_state_decoder = {
        'best_val_loss': float('inf'),
        'epochs_no_improve': 0,
        'early_stop_flag': False,
        'patience': patience_decoder,
        'delta': delta_decoder,
        'checkpoint_path': checkpoint_path_decoder # Full path
    }

    for epoch in range(num_epochs_decoder):
        if early_stopping_state_decoder['early_stop_flag']:
            print(f"JEPA State Decoder early stopping triggered before epoch {epoch+1}. Exiting decoder training loop.")
            break

        print(f"\n--- JEPA State Decoder Epoch {epoch+1}/{num_epochs_decoder} ---")
        if jepa_model: jepa_model.eval() # Main JEPA model provides embeddings
        jepa_decoder_model.train()

        epoch_loss_train = 0
        num_batches_train = len(train_dataloader) if train_dataloader else 0

        if num_batches_train == 0:
            print(f"JEPA Decoder Epoch {epoch+1} has no training data. Skipping training phase.")
        else:
            for batch_idx, (s_t, a_t, _, s_t_plus_1) in enumerate(train_dataloader):
                s_t, s_t_plus_1 = s_t.to(device), s_t_plus_1.to(device)
                if action_type == 'discrete':
                    a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device) if a_t.ndim == 1 else F.one_hot(a_t.long(), num_classes=action_dim).float().to(device)
                    if a_t_processed.shape[0] != s_t.shape[0]:
                         a_t_processed = F.one_hot(a_t.long().view(-1), num_classes=action_dim).float().to(device)
                else:
                    a_t_processed = a_t.float().to(device)

            if batch_idx == 0 and epoch == 0: # epoch here is 0-indexed
                print(f"Debug Devices (JEPA State Decoder Epoch):")
                print(f"  s_t device: {s_t.device}")
                print(f"  a_t_processed device: {a_t_processed.device}")
                print(f"  s_t_plus_1 device: {s_t_plus_1.device}")
                print(f"  jepa_decoder_model device (first param): {next(jepa_decoder_model.parameters()).device}")
                if jepa_model:
                    print(f"  jepa_model (base for decoder) device (first param): {next(jepa_model.parameters()).device}")

                optimizer_jepa_decoder.zero_grad()
                with torch.no_grad():
                    if not jepa_model: # Should be caught by caller, but good to check
                        print("Error: Main JEPA model is None for JEPA decoder training.")
                        early_stopping_state_decoder['early_stop_flag'] = True; break
                    pred_emb, _, _, _ = jepa_model(s_t, a_t_processed, s_t_plus_1)

                jepa_predictor_output = pred_emb.detach()
                reconstructed_s_t_plus_1 = jepa_decoder_model(jepa_predictor_output)
                loss = loss_fn(reconstructed_s_t_plus_1, s_t_plus_1)
                loss.backward()
                optimizer_jepa_decoder.step()
                epoch_loss_train += loss.item()

                if (batch_idx + 1) % decoder_log_interval == 0:
                    print(f"  JEPA Decoder Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches_train}: Train Loss {loss.item():.4f}")
            if early_stopping_state_decoder['early_stop_flag']: break # From error in batch loop

        avg_train_loss = epoch_loss_train / num_batches_train if num_batches_train > 0 else 0
        print(f"  Avg Train JEPA Decoder L (Epoch {epoch+1}): {avg_train_loss:.4f}")

        # Validation Phase
        if val_dataloader:
            jepa_decoder_model.eval()
            if jepa_model: jepa_model.eval()

            epoch_loss_val = 0
            num_batches_val = len(val_dataloader)
            with torch.no_grad():
                for val_batch_idx, (s_t_val, a_t_val, _, s_t_plus_1_val) in enumerate(val_dataloader):
                    s_t_val, s_t_plus_1_val = s_t_val.to(device), s_t_plus_1_val.to(device)
                    if action_type == 'discrete':
                        a_t_val_processed = F.one_hot(a_t_val.long().view(-1), num_classes=action_dim).float().to(device) if a_t_val.ndim == 1 else F.one_hot(a_t_val.long(), num_classes=action_dim).float().to(device)
                        if a_t_val_processed.shape[0] != s_t_val.shape[0]:
                           a_t_val_processed = F.one_hot(a_t_val.long().view(-1), num_classes=action_dim).float().to(device)
                    else:
                        a_t_val_processed = a_t_val.float().to(device)

                    if not jepa_model:
                         print("Error: Main JEPA model is None during JEPA decoder validation.")
                         early_stopping_state_decoder['early_stop_flag'] = True; break
                    pred_emb_val, _, _, _ = jepa_model(s_t_val, a_t_val_processed, s_t_plus_1_val)
                    jepa_predictor_output_val = pred_emb_val.detach()
                    reconstructed_s_t_plus_1_val = jepa_decoder_model(jepa_predictor_output_val)
                    val_loss = loss_fn(reconstructed_s_t_plus_1_val, s_t_plus_1_val)
                    epoch_loss_val += val_loss.item()

                    # Plotting logic
                    if val_batch_idx == 0 and decoder_training_config.get('enable_validation_plot', False):
                        os.makedirs(validation_plot_dir_full, exist_ok=True)
                        num_plot_samples = min(4, s_t_val.shape[0])
                        random_indices = np.random.choice(s_t_val.shape[0], num_plot_samples, replace=False) if s_t_val.shape[0] > num_plot_samples else range(s_t_val.shape[0])
                        for i in random_indices:
                            true_img = s_t_plus_1_val[i].cpu().numpy()
                            pred_img = reconstructed_s_t_plus_1_val[i].cpu().numpy()
                            if true_img.shape[0] == 1 or true_img.shape[0] == 3: # C, H, W
                                true_img = np.transpose(true_img, (1, 2, 0))
                                pred_img = np.transpose(pred_img, (1, 2, 0))
                            if true_img.shape[-1] == 1: # Grayscale, squeeze channel
                                true_img = true_img.squeeze(axis=2)
                                pred_img = pred_img.squeeze(axis=2)
                            if true_img.dtype == np.float32 or true_img.dtype == np.float64: # Clip if float
                                true_img = np.clip(true_img, 0, 1)
                                pred_img = np.clip(pred_img, 0, 1)
                            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                            axes[0].imshow(true_img); axes[0].set_title("True Image"); axes[0].axis('off')
                            axes[1].imshow(pred_img); axes[1].set_title("Predicted Image"); axes[1].axis('off')
                            plot_filename = os.path.join(validation_plot_dir_full, f"epoch_{epoch+1}_valbatch_{val_batch_idx}_sample_{i}.png")
                            plt.savefig(plot_filename); plt.close(fig)
                        print(f"  JEPA Decoder: Saved {num_plot_samples} validation image samples to {validation_plot_dir_full}")
                if early_stopping_state_decoder['early_stop_flag']: break

                avg_val_loss = epoch_loss_val / num_batches_val if num_batches_val > 0 else float('inf')
                print(f"--- JEPA Decoder Epoch {epoch+1} Validation Summary ---")
                print(f"  Avg Val JEPA Decoder L: {avg_val_loss:.4f}")

                if avg_val_loss < early_stopping_state_decoder['best_val_loss'] - early_stopping_state_decoder['delta']:
                    early_stopping_state_decoder['best_val_loss'] = avg_val_loss
                    if early_stopping_state_decoder['checkpoint_path']:
                        os.makedirs(os.path.dirname(early_stopping_state_decoder['checkpoint_path']), exist_ok=True)
                        torch.save(jepa_decoder_model.state_dict(), early_stopping_state_decoder['checkpoint_path'])
                    early_stopping_state_decoder['epochs_no_improve'] = 0
                    print(f"  JEPA Decoder: Val loss improved. Saved model to {early_stopping_state_decoder['checkpoint_path']}")
                else:
                    early_stopping_state_decoder['epochs_no_improve'] += 1
                    print(f"  JEPA Decoder: No val improvement for {early_stopping_state_decoder['epochs_no_improve']} epochs.")
                    if early_stopping_state_decoder['epochs_no_improve'] >= early_stopping_state_decoder['patience']:
                        early_stopping_state_decoder['early_stop_flag'] = True
                        print("  JEPA Decoder: Early stopping triggered.")
        else: # No validation dataloader
            print(f"--- JEPA Decoder Epoch {epoch+1} Training Summary (No Validation) ---")
            if early_stopping_state_decoder['checkpoint_path']: # Save last epoch if no validation
                os.makedirs(os.path.dirname(early_stopping_state_decoder['checkpoint_path']), exist_ok=True)
                torch.save(jepa_decoder_model.state_dict(), early_stopping_state_decoder['checkpoint_path'])
                print(f"  JEPA Decoder: Saved model from last epoch to {early_stopping_state_decoder['checkpoint_path']} (no validation set)")

        if early_stopping_state_decoder['early_stop_flag']: break

    print("JEPA State Decoder training finished.")

    # Load the best model if it was saved
    best_checkpoint_file = early_stopping_state_decoder['checkpoint_path']
    if best_checkpoint_file and os.path.exists(best_checkpoint_file):
        print(f"Loading best JEPA State Decoder model from {best_checkpoint_file}")
        jepa_decoder_model.load_state_dict(torch.load(best_checkpoint_file, map_location=device))
        return best_checkpoint_file
    elif best_checkpoint_file: # Path was set, but no file (e.g. no training/val epochs ran or never improved)
         print(f"JEPA State Decoder checkpoint {best_checkpoint_file} not found. Using model state as is.")
         return None
    return None


def run_training_epochs(
    models_map, optimizers_map, losses_map, dataloaders_map,
    device, config, action_dim, action_type,
    image_h_w, input_channels, # For reward MLP input calculation if needed
    std_enc_dec_loaded_successfully=False, # New argument
    jepa_loaded_successfully=False         # New argument
):
    # Unpack from maps/config for convenience
    std_enc_dec = models_map.get('std_enc_dec')
    jepa_model = models_map.get('jepa')
    # Reward MLPs and JEPA decoder are handled later, focus on main model refactor first
    reward_mlp_enc_dec = models_map.get('reward_mlp_enc_dec')
    reward_mlp_jepa = models_map.get('reward_mlp_jepa')
    jepa_decoder = models_map.get('jepa_decoder') # New model

    optimizer_std_enc_dec = optimizers_map.get('std_enc_dec')
    optimizer_jepa = optimizers_map.get('jepa')
    optimizer_reward_mlp_enc_dec = optimizers_map.get('reward_mlp_enc_dec')
    optimizer_reward_mlp_jepa = optimizers_map.get('reward_mlp_jepa')
    optimizer_jepa_decoder = optimizers_map.get('jepa_decoder') # New optimizer

    mse_loss_fn = losses_map['mse']
    aux_loss_fn = losses_map.get('aux_fn')
    aux_loss_name = losses_map.get('aux_name', "None")
    aux_loss_weight = losses_map.get('aux_weight', 0.0)

    train_dataloader = dataloaders_map['train']
    val_dataloader = dataloaders_map.get('val')

    # Configs
    early_stopping_config = config.get('early_stopping', {})
    model_dir = config.get('model_dir', 'trained_models/')
    # Configs
    early_stopping_config = config.get('early_stopping', {})
    model_dir = config.get('model_dir', 'trained_models/')
    # General early stopping params, can be overridden per model if needed
    patience = early_stopping_config.get('patience', 10)
    delta = early_stopping_config.get('delta', 0.001)

    # Checkpoint paths
    checkpoint_path_enc_dec = os.path.join(model_dir, early_stopping_config.get('checkpoint_path_enc_dec', 'best_encoder_decoder.pth'))
    checkpoint_path_jepa = os.path.join(model_dir, early_stopping_config.get('checkpoint_path_jepa', 'best_jepa.pth'))

    enc_dec_mlp_config = config.get('reward_predictors', {}).get('encoder_decoder_reward_mlp', {}) # For reward MLP section
    jepa_mlp_config = config.get('reward_predictors', {}).get('jepa_reward_mlp', {}) # For reward MLP section

    num_epochs = config.get('num_epochs', 10)
    log_interval = config.get('log_interval', 50)

    print(f"Starting training, main models for up to {num_epochs} epochs...")

    # Initialize early_stopping_state dictionaries
    early_stopping_state_enc_dec = {
        'best_val_loss': float('inf'),
        'epochs_no_improve': 0,
        'early_stop_flag': not (std_enc_dec and optimizer_std_enc_dec), # Pre-stop if model/opt missing
        'patience': early_stopping_config.get('patience_enc_dec', patience), # Specific or general patience
        'delta': early_stopping_config.get('delta_enc_dec', delta),         # Specific or general delta
        'checkpoint_path': checkpoint_path_enc_dec
    }
    early_stopping_state_jepa = {
        'best_val_loss': float('inf'),
        'epochs_no_improve': 0,
        'early_stop_flag': not (jepa_model and optimizer_jepa), # Pre-stop if model/opt missing
        'patience': early_stopping_config.get('patience_jepa', patience),
        'delta': early_stopping_config.get('delta_jepa', delta),
        'checkpoint_path': checkpoint_path_jepa
    }

    # Handle Skip Training Flags based on loaded models
    training_options = config.get('training_options', {})
    # Handle Skip Training Flags based on loaded models
    training_options = config.get('training_options', {})
    skip_std_enc_dec_opt = training_options.get('skip_std_enc_dec_training_if_loaded', False)
    skip_jepa_opt = training_options.get('skip_jepa_training_if_loaded', False)

    if std_enc_dec_loaded_successfully and skip_std_enc_dec_opt and not early_stopping_state_enc_dec['early_stop_flag']:
        early_stopping_state_enc_dec['early_stop_flag'] = True
        print("Standard Encoder/Decoder training will be skipped as a pre-trained model was loaded and skip option is enabled.")

    if jepa_loaded_successfully and skip_jepa_opt and not early_stopping_state_jepa['early_stop_flag']:
        early_stopping_state_jepa['early_stop_flag'] = True
        print("JEPA model training will be skipped as a pre-trained model was loaded and skip option is enabled.")


    for epoch in range(num_epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{num_epochs} ---")

        # Track epoch losses for summary
        epoch_loss_std_train_primary = 0
        epoch_loss_jepa_train_primary = 0
        epoch_loss_jepa_train_aux = 0 # Only for JEPA

        # --- Standard Encoder/Decoder Training ---
        if std_enc_dec and not early_stopping_state_enc_dec['early_stop_flag']:
            print(f"Running training and validation for Standard Encoder/Decoder (Epoch {epoch+1})...")
            early_stopping_state_enc_dec, train_loss_std, _ = _train_validate_model_epoch(
                model=std_enc_dec,
                optimizer=optimizer_std_enc_dec,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                loss_fn=mse_loss_fn,
                aux_loss_fn=None, # No aux loss for std_enc_dec
                aux_loss_name="N/A",
                aux_loss_weight=0,
                device=device,
                epoch_num=epoch + 1,
                log_interval=log_interval,
                action_dim=action_dim,
                action_type=action_type,
                early_stopping_state=early_stopping_state_enc_dec,
                checkpoint_path=early_stopping_state_enc_dec['checkpoint_path'],
                model_name_log_prefix="StdEncDec"
            )
            epoch_loss_std_train_primary = train_loss_std
        elif not std_enc_dec:
             if not early_stopping_state_enc_dec['early_stop_flag']: print(f"StdEncDec model not provided, skipping epoch {epoch+1}.")
             early_stopping_state_enc_dec['early_stop_flag'] = True # Ensure it's marked as stopped
        elif early_stopping_state_enc_dec['early_stop_flag']:
            print(f"StdEncDec training already early stopped or skipped. Skipping epoch {epoch+1}.")


        # --- JEPA Model Training ---
        if jepa_model and not early_stopping_state_jepa['early_stop_flag']:
            print(f"Running training and validation for JEPA Model (Epoch {epoch+1})...")
            early_stopping_state_jepa, train_loss_jepa_pred, train_loss_jepa_aux_raw = _train_validate_model_epoch(
                model=jepa_model,
                optimizer=optimizer_jepa,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                loss_fn=mse_loss_fn, # Primary loss for JEPA (prediction)
                aux_loss_fn=aux_loss_fn,
                aux_loss_name=aux_loss_name,
                aux_loss_weight=aux_loss_weight,
                device=device,
                epoch_num=epoch + 1,
                log_interval=log_interval,
                action_dim=action_dim,
                action_type=action_type,
                early_stopping_state=early_stopping_state_jepa,
                checkpoint_path=early_stopping_state_jepa['checkpoint_path'],
                model_name_log_prefix="JEPA",
                update_target_fn=lambda: jepa_model.perform_ema_update() # Pass target update function
            )
            epoch_loss_jepa_train_primary = train_loss_jepa_pred
            epoch_loss_jepa_train_aux = train_loss_jepa_aux_raw # This is raw aux, will be weighted for total summary
        elif not jepa_model:
            if not early_stopping_state_jepa['early_stop_flag']: print(f"JEPA model not provided, skipping epoch {epoch+1}.")
            early_stopping_state_jepa['early_stop_flag'] = True # Ensure it's marked as stopped
        elif early_stopping_state_jepa['early_stop_flag']:
            print(f"JEPA training already early stopped or skipped. Skipping epoch {epoch+1}.")


        # --- Epoch Summary for Main Models ---
        print(f"--- Epoch {epoch+1}/{num_epochs} Overall Training Summary ---")
        if std_enc_dec and not early_stopping_state_enc_dec['early_stop_flag']: # Log if it ran this epoch or was just stopped
             print(f"  Avg Train StdEncDec L: {epoch_loss_std_train_primary:.4f}")
        elif std_enc_dec and early_stopping_state_enc_dec['early_stop_flag'] and epoch_loss_std_train_primary == 0 : # It was stopped before this epoch
             print(f"  StdEncDec training was already stopped/skipped.")

        if jepa_model and not early_stopping_state_jepa['early_stop_flag']:
            avg_total_jepa_train_loss = epoch_loss_jepa_train_primary + (epoch_loss_jepa_train_aux * aux_loss_weight if aux_loss_fn and aux_loss_weight > 0 else 0)
            print(f"  Avg Train JEPA Pred L: {epoch_loss_jepa_train_primary:.4f}, "
                  f"{aux_loss_name} AuxRawL: {epoch_loss_jepa_train_aux:.4f}, "
                  f"Total Train JEPA L: {avg_total_jepa_train_loss:.4f}")
        elif jepa_model and early_stopping_state_jepa['early_stop_flag'] and epoch_loss_jepa_train_primary == 0:
            print(f"  JEPA training was already stopped/skipped.")


        # --- Check for breaking main epoch loop ---
        if early_stopping_state_enc_dec['early_stop_flag'] and early_stopping_state_jepa['early_stop_flag']:
            print("Both main models (StdEncDec, JEPA) have triggered early stopping or were skipped. Proceeding to subsequent training stages if any.")
            break # Break from main model epoch loop

    print("Main models training loop finished.")

    # --- Reward MLP Training Loop ---
    # (This section needs to be reviewed to ensure it uses the potentially updated models correctly after the loop)
    # For now, assume it runs after the main model loop completes or early stops.
    # The key is that std_enc_dec and jepa_model (if loaded from checkpoint) should be the "best" versions.

    # Load best models if they were saved via early stopping, for subsequent Reward MLP and JEPA Decoder training
    if std_enc_dec and early_stopping_state_enc_dec['checkpoint_path'] and os.path.exists(early_stopping_state_enc_dec['checkpoint_path']):
        print(f"Loading best Standard Encoder/Decoder model from {early_stopping_state_enc_dec['checkpoint_path']} for subsequent tasks.")
        std_enc_dec.load_state_dict(torch.load(early_stopping_state_enc_dec['checkpoint_path'], map_location=device))
    if jepa_model and early_stopping_state_jepa['checkpoint_path'] and os.path.exists(early_stopping_state_jepa['checkpoint_path']):
        print(f"Loading best JEPA model from {early_stopping_state_jepa['checkpoint_path']} for subsequent tasks.")
        jepa_model.load_state_dict(torch.load(early_stopping_state_jepa['checkpoint_path'], map_location=device))


    # (Original Reward MLP and JEPA State Decoder training loops follow)
    # Ensure these loops use the potentially reloaded best models.
    # The rest of the code for Reward MLP and JEPA State Decoder training will remain here.
    # --- Reward MLP Training using _train_reward_mlp_epoch ---
    if reward_mlp_enc_dec and enc_dec_mlp_config.get('enabled', False) and optimizer_reward_mlp_enc_dec and std_enc_dec and train_dataloader:
        _train_reward_mlp_epoch(
            reward_mlp_model=reward_mlp_enc_dec,
            base_model=std_enc_dec, # This is the best loaded std_enc_dec
            optimizer_reward_mlp=optimizer_reward_mlp_enc_dec,
            train_dataloader=train_dataloader,
            loss_fn=mse_loss_fn,
            device=device,
            action_dim=action_dim,
            action_type=action_type,
            model_name_log_prefix="Reward MLP (Enc-Dec)",
            num_epochs_reward_mlp=enc_dec_mlp_config.get('num_epochs', 1), # Default to 1 epoch if not specified
            log_interval_reward_mlp=enc_dec_mlp_config.get('log_interval', log_interval), # Use specific or general log_interval
            is_jepa_base_model=False
        )
    elif enc_dec_mlp_config.get('enabled', False):
        print("Reward MLP (Enc-Dec) training skipped due to missing components (model, optimizer, base_model, or dataloader).")

    if reward_mlp_jepa and jepa_mlp_config.get('enabled', False) and optimizer_reward_mlp_jepa and jepa_model and train_dataloader:
        _train_reward_mlp_epoch(
            reward_mlp_model=reward_mlp_jepa,
            base_model=jepa_model, # This is the best loaded jepa_model
            optimizer_reward_mlp=optimizer_reward_mlp_jepa,
            train_dataloader=train_dataloader,
            loss_fn=mse_loss_fn,
            device=device,
            action_dim=action_dim,
            action_type=action_type,
            model_name_log_prefix="Reward MLP (JEPA)",
            num_epochs_reward_mlp=jepa_mlp_config.get('num_epochs', 1), # Default to 1 epoch
            log_interval_reward_mlp=jepa_mlp_config.get('log_interval', log_interval),
            is_jepa_base_model=True
        )
    elif jepa_mlp_config.get('enabled', False):
        print("Reward MLP (JEPA) training skipped due to missing components (model, optimizer, base_model, or dataloader).")

    # --- JEPA State Decoder Training Loop ---
    # The JEPA State Decoder training loop remains as is, but it will use the
    # potentially best JEPA model loaded above.
    # --- JEPA State Decoder Training ---
    final_checkpoint_path_jepa_decoder = None
    jepa_decoder_training_config = config.get('jepa_decoder_training', {})

    if (jepa_decoder and optimizer_jepa_decoder and jepa_model and
        jepa_decoder_training_config.get('enabled', False) and train_dataloader):
        # Note: jepa_model here is the one loaded from its best checkpoint (if saved)
        final_checkpoint_path_jepa_decoder = _train_jepa_state_decoder(
            jepa_decoder_model=jepa_decoder,
            jepa_model=jepa_model,
            optimizer_jepa_decoder=optimizer_jepa_decoder,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_fn=mse_loss_fn,
            device=device,
            action_dim=action_dim,
            action_type=action_type,
            decoder_training_config=jepa_decoder_training_config,
            main_model_dir=model_dir, # Pass the main model directory for path construction
            general_log_interval=log_interval # Pass general log interval as fallback
        )
    elif jepa_decoder_training_config.get('enabled', False):
        print("JEPA State Decoder training skipped due to missing components (decoder model, its optimizer, main JEPA model, or dataloader).")
    else:
        if not jepa_decoder: print("JEPA State Decoder model not provided.")
        elif not optimizer_jepa_decoder: print("Optimizer for JEPA State Decoder not provided.")
        elif not jepa_model: print("Main JEPA model (for embeddings) not available for JEPA State Decoder training.")
        elif not jepa_decoder_training_config.get('enabled', False): print("JEPA State Decoder training is disabled in config.")
        elif not train_dataloader: print("Train dataloader not available for JEPA State Decoder training.")


    print("\nAll training processes finished from training_engine.")

    final_checkpoint_enc_dec = early_stopping_state_enc_dec.get('checkpoint_path') \
        if std_enc_dec and early_stopping_state_enc_dec.get('checkpoint_path') and os.path.exists(early_stopping_state_enc_dec['checkpoint_path']) \
        else None
    final_checkpoint_jepa = early_stopping_state_jepa.get('checkpoint_path') \
        if jepa_model and early_stopping_state_jepa.get('checkpoint_path') and os.path.exists(early_stopping_state_jepa['checkpoint_path']) \
        else None
    # final_checkpoint_path_jepa_decoder is already assigned from the helper function's return

    return {
        "best_checkpoint_enc_dec": final_checkpoint_enc_dec,
        "best_checkpoint_jepa": final_checkpoint_jepa,
        "best_checkpoint_jepa_decoder": final_checkpoint_path_jepa_decoder
    }
