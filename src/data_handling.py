# Contents for src/data_handling.py
import torch
from torch.utils.data import DataLoader
from src.utils.data_utils import collect_random_episodes, collect_ppo_episodes # Absolute import from src

def prepare_dataloaders(config, validation_split):
    print("Starting data collection...")
    key_max_steps = 'max_steps_per_episode_data_collection'
    image_h_w = config['image_size']
    img_size_tuple = (image_h_w, image_h_w)

    ppo_config = config.get('ppo_agent', {})
    use_ppo = ppo_config.get('enabled', False)

    if use_ppo:
        print("Using PPO agent for data collection.")
        train_dataset, val_dataset = collect_ppo_episodes(
            config=config, # Pass the full config object
            max_steps_per_episode=config.get(key_max_steps, 200),
            image_size=img_size_tuple,
            validation_split_ratio=validation_split
        )
    else:
        print("Using random actions for data collection.")
        train_dataset, val_dataset = collect_random_episodes(
            config=config, # Pass the full config object
            max_steps_per_episode=config.get(key_max_steps, 200),
            image_size=img_size_tuple,
            validation_split_ratio=validation_split
        )

    if len(train_dataset) == 0:
        print("No training data collected. Exiting program or handling as error.")
        return None, None

    if validation_split > 0 and len(val_dataset) == 0:
        print("Warning: Validation split is > 0 but no validation data was collected. Check dataset size and split ratio.")


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_dataloader = None
    if len(val_dataset) > 0:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True if torch.cuda.is_available() else False
        )

    return train_dataloader, val_dataloader
