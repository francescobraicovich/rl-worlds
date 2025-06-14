import torch
import os # For path joining and checking existence
import wandb
import time

# Import functions from the new modules in the 'src' directory
from src.utils.config_utils import load_config
from src.utils.env_utils import get_env_details
from src.data_handling import prepare_dataloaders
from src.model_setup import initialize_models
from src.loss_setup import initialize_loss_functions
from src.optimizer_setup import initialize_optimizers
from src.training_engine import run_training_epochs

def main():
    # 1. Load Configuration
    config = load_config(config_path='config.yaml')

    # Initialize wandb
    wandb_config = config.get('wandb', {})
    wandb_run = None
    if wandb_config.get('enabled', False):
        try:
            wandb_run = wandb.init(
                project=wandb_config.get('project'),
                entity=wandb_config.get('entity'),
                name=f"{wandb_config.get('run_name_prefix', 'exp')}-{time.strftime('%Y%m%d-%H%M%S')}",
                config=config  # Log the entire experiment config
            )
            print("Weights & Biases initialized successfully.")
        except Exception as e:
            print(f"Error initializing Weights & Biases: {e}. Proceeding without W&B.")
            wandb_run = None # Ensure wandb_run is None if init fails
    else:
        print("Weights & Biases is disabled in the configuration.")


    # 2. Setup Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    model_dir = config.get('model_dir', 'trained_models/')
    dataset_dir = config.get('dataset_dir', 'datasets/')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"Ensured model directory exists: {model_dir}")
    print(f"Ensured dataset directory exists: {dataset_dir}")

    # 3. Get Environment Details
    print("Fetching environment details...")
    action_dim, action_type, observation_space = get_env_details(config['environment_name'])

    # 4. Prepare Dataloaders
    early_stopping_config = config.get('early_stopping', {})
    validation_split = early_stopping_config.get('validation_split', 0.2)
    dataloaders_map = {}
    train_dataloader, val_dataloader = prepare_dataloaders(config, validation_split)
    if train_dataloader is None:
        print("Exiting due to no training data.")
        return
    dataloaders_map['train'] = train_dataloader
    if val_dataloader:
        dataloaders_map['val'] = val_dataloader

    # 5. Initialize Models
    image_h_w = config['image_size']
    input_channels = config.get('input_channels', 3)
    models_map = initialize_models(config, action_dim, device, image_h_w, input_channels)

    # --- Model Loading Logic ---
    load_model_path = config.get('load_model_path', '')
    model_type_to_load = config.get('model_type_to_load', 'std_enc_dec') # e.g., 'std_enc_dec', 'jepa'

    std_enc_dec_loaded_successfully = False
    jepa_model_loaded_successfully = False

    if load_model_path:
        full_model_load_path = os.path.join(model_dir, load_model_path)
        if os.path.exists(full_model_load_path):
            print(f"Attempting to load pre-trained model from: {full_model_load_path}")
            try:
                if model_type_to_load == 'std_enc_dec' and models_map.get('std_enc_dec'):
                    models_map['std_enc_dec'].load_state_dict(torch.load(full_model_load_path, map_location=device))
                    print(f"Loaded Standard Encoder/Decoder model from {full_model_load_path}")
                    std_enc_dec_loaded_successfully = True
                elif model_type_to_load == 'jepa' and models_map.get('jepa'):
                    models_map['jepa'].load_state_dict(torch.load(full_model_load_path, map_location=device))
                    print(f"Loaded JEPA model from {full_model_load_path}")
                    jepa_model_loaded_successfully = True
                # Add other model types here if they can be loaded independently via 'load_model_path'
                else:
                    print(f"Warning: Model type '{model_type_to_load}' specified for loading, but it's either not configured or not supported by this loading block.")
            except Exception as e:
                print(f"Error loading model {model_type_to_load} from {full_model_load_path}: {e}. Proceeding with default initialization for this model.")
        else:
            print(f"Warning: Specified model path '{full_model_load_path}' not found. Proceeding with default initialization(s).")
    else:
        print("No pre-trained model path specified in 'load_model_path'. Models will be initialized from scratch or use their default initialization.")
    # --- End of Model Loading Logic ---

    std_enc_dec = models_map.get('std_enc_dec')
    jepa_model = models_map.get('jepa')

    jepa_model_latent_dim_for_dino = None
    if jepa_model and config.get('auxiliary_loss', {}).get('type') == 'dino':
        jepa_model_latent_dim_for_dino = jepa_model.latent_dim

    losses_map = initialize_loss_functions(config, device, jepa_model_latent_dim=jepa_model_latent_dim_for_dino)
    optimizers_map = initialize_optimizers(models_map, config)

    # 8. Run Training Epochs
    training_results = run_training_epochs( # Store results
        models_map=models_map,
        optimizers_map=optimizers_map,
        losses_map=losses_map,
        dataloaders_map=dataloaders_map,
        device=device,
        config=config,
        action_dim=action_dim,
        action_type=action_type,
        image_h_w=image_h_w,
        input_channels=input_channels,
        std_enc_dec_loaded_successfully=std_enc_dec_loaded_successfully, # Pass flag
        jepa_loaded_successfully=jepa_model_loaded_successfully,          # Pass flag
        wandb_run=wandb_run # Pass wandb_run object
    )

    # 9. Post-Training: Load best models and set to eval mode
    # Checkpoint paths from config (used by training_engine for saving, here for loading)
    # model_dir is available. training_results contains paths to best models saved by engine.

    print("\nLoading best models (if available) after training and setting to eval mode...")

    # Standard Encoder/Decoder
    if std_enc_dec: # Check if model was initialized
        best_checkpoint_enc_dec_path = training_results.get("best_checkpoint_enc_dec")
        if best_checkpoint_enc_dec_path and os.path.exists(best_checkpoint_enc_dec_path):
            print(f"Loading best Encoder/Decoder model from {best_checkpoint_enc_dec_path}")
            std_enc_dec.load_state_dict(torch.load(best_checkpoint_enc_dec_path, map_location=device))
        elif not std_enc_dec_loaded_successfully: # Only print if not initially loaded
             print(f"No best checkpoint found for Encoder/Decoder at expected path. Model remains in its last training state.")
        std_enc_dec.eval()

    # JEPA Model
    if jepa_model: # Check if model was initialized
        best_checkpoint_jepa_path = training_results.get("best_checkpoint_jepa")
        if best_checkpoint_jepa_path and os.path.exists(best_checkpoint_jepa_path):
            print(f"Loading best JEPA model from {best_checkpoint_jepa_path}")
            jepa_model.load_state_dict(torch.load(best_checkpoint_jepa_path, map_location=device))
        elif not jepa_model_loaded_successfully: # Only print if not initially loaded
            print(f"No best checkpoint found for JEPA at expected path. Model remains in its last training state.")
        jepa_model.eval()

    # JEPA State Decoder
    jepa_decoder = models_map.get('jepa_decoder')
    jepa_decoder_training_config = config.get('jepa_decoder_training', {})
    if jepa_decoder and jepa_decoder_training_config.get('enabled', False):
        # Path to best decoder model is now returned by run_training_epochs
        best_checkpoint_jepa_decoder_path = training_results.get("best_checkpoint_jepa_decoder")
        print(f"Attempting to load best JEPA State Decoder (if available) after training...")
        if best_checkpoint_jepa_decoder_path and os.path.exists(best_checkpoint_jepa_decoder_path):
            print(f"Loading best JEPA State Decoder model from {best_checkpoint_jepa_decoder_path}")
            jepa_decoder.load_state_dict(torch.load(best_checkpoint_jepa_decoder_path, map_location=device))
        else:
            # This message is important if decoder training was enabled but no checkpoint was saved/found
            print(f"No best checkpoint found for JEPA State Decoder at expected path. Model remains in its last training state (if any training occurred).")
        jepa_decoder.eval()
    elif jepa_decoder : # Decoder exists but was not enabled for training
        print("JEPA State Decoder was initialized but not enabled for training. Setting to eval mode.")
        jepa_decoder.eval()


    # Reward MLPs
    if models_map.get('reward_mlp_enc_dec'):
        models_map['reward_mlp_enc_dec'].eval()
        print("Encoder-Decoder Reward MLP set to eval mode.")
    if models_map.get('reward_mlp_jepa'):
        models_map['reward_mlp_jepa'].eval()
        print("JEPA Reward MLP set to eval mode.")

    if losses_map.get('aux_fn') and hasattr(losses_map['aux_fn'], 'eval'):
        losses_map['aux_fn'].eval()
        print("Auxiliary loss function (if DINO) set to eval mode.")

    print("\nProcess complete. Relevant models are in eval mode.")

    if wandb_run:
        wandb_run.finish()
        print("Weights & Biases run finished.")

if __name__ == '__main__':
    main()
