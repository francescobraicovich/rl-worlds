import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import random  # Added for shuffling episodes
import os
import pickle
from stable_baselines3 import PPO
from src.rl_agent import create_ppo_agent, train_ppo_agent
# from stable_baselines3.common.vec_env import DummyVecEnv # Not strictly needed here if rl_agent handles it
# import cv2 # For image resizing - Removed as torchvision.transforms is used


class ExperienceDataset(Dataset):
    def __init__(self, states, actions, rewards, next_states, transform=None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.transform = transform

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_state = self.next_states[idx]

        if self.transform:
            state = self.transform(state)
            next_state = self.transform(next_state)

        # Convert action to tensor, ensure it's float for potential nn.Linear embedding
        # Adjust dtype based on action type (discrete typically long, continuous float)
        # For simplicity, let's assume actions will be made float.
        # If discrete, they might be indices; ensure they are handled appropriately later (e.g. one-hot or embedding layer).
        action_tensor = torch.tensor(action, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        return state, action_tensor, reward_tensor, next_state


def collect_random_episodes(config, max_steps_per_episode, image_size, validation_split_ratio):
    env_name = config['environment_name']
    num_episodes = config['num_episodes_data_collection']

    # New configuration keys
    dataset_dir = config['dataset_dir']
    load_dataset_filename_config = config['load_dataset_path'] # Filename or path relative to dataset_dir
    save_dataset_filename_config = config['dataset_filename'] # Filename for saving new datasets

    os.makedirs(dataset_dir, exist_ok=True)

    # Determine save filename with .npz extension
    save_file_name_with_ext = save_dataset_filename_config
    if not save_file_name_with_ext.lower().endswith('.npz'):
        save_file_name_with_ext = os.path.splitext(save_file_name_with_ext)[0] + '.npz'

    # Determine load filename with .npz extension (if load_dataset_filename is provided)
    load_file_name_with_ext = None
    if load_dataset_filename_config:
        load_file_name_with_ext = load_dataset_filename_config
        if not load_file_name_with_ext.lower().endswith('.npz'):
            load_file_name_with_ext = os.path.splitext(load_file_name_with_ext)[0] + '.npz'

    data_loaded_successfully = False
    if load_file_name_with_ext:
        dataset_path_to_load = os.path.join(dataset_dir, load_file_name_with_ext)
        if os.path.exists(dataset_path_to_load):
            print(f"Loading dataset from {dataset_path_to_load}...")
            try:
                loaded_data = np.load(dataset_path_to_load, allow_pickle=True)

                metadata_pickled_array = loaded_data['metadata']
                if metadata_pickled_array.ndim == 0 or metadata_pickled_array.size == 1:
                    metadata = pickle.loads(metadata_pickled_array.item())
                else:
                    raise ValueError("Metadata format error in .npz file.")

                loaded_env_name = metadata.get('environment_name')
                if loaded_env_name != env_name:
                    print(f"Error: Mismatch between loaded dataset environment ('{loaded_env_name}') and config environment ('{env_name}').")
                    raise ValueError("Environment mismatch in loaded dataset.")

                # For random collection, no specific collection_method check is strictly needed here,
                # but you could add one if you want to differentiate between 'random' and 'ppo' loaded datasets.

                preprocess = T.Compose([
                    T.ToPILImage(),
                    T.Resize(image_size),
                    T.ToTensor()
                ])

                def get_list_from_npz(npz_file, key):
                    if key not in npz_file or npz_file[key].size == 0:
                        return []
                    return list(npz_file[key])

                train_states_loaded = get_list_from_npz(loaded_data, 'train_states')
                train_actions_loaded = get_list_from_npz(loaded_data, 'train_actions')
                train_rewards_loaded = get_list_from_npz(loaded_data, 'train_rewards')
                train_next_states_loaded = get_list_from_npz(loaded_data, 'train_next_states')

                val_states_loaded = get_list_from_npz(loaded_data, 'val_states')
                val_actions_loaded = get_list_from_npz(loaded_data, 'val_actions')
                val_rewards_loaded = get_list_from_npz(loaded_data, 'val_rewards')
                val_next_states_loaded = get_list_from_npz(loaded_data, 'val_next_states')

                loaded_train_dataset = ExperienceDataset(
                    train_states_loaded, train_actions_loaded, train_rewards_loaded, train_next_states_loaded,
                    transform=preprocess
                )
                loaded_val_dataset = ExperienceDataset(
                    val_states_loaded, val_actions_loaded, val_rewards_loaded, val_next_states_loaded,
                    transform=preprocess
                )

                print(f"Successfully loaded dataset for environment '{loaded_env_name}' ... from {dataset_path_to_load}.")
                data_loaded_successfully = True
                return loaded_train_dataset, loaded_val_dataset
            except Exception as e:
                print(f"Error loading dataset from {dataset_path_to_load}: {e}. Proceeding to data collection.")
        else:
            print(f"Warning: Dataset {dataset_path_to_load} not found. Proceeding to data collection.")
    else:
        print("`load_dataset_path` (or its .npz version) is effectively empty or not found. Proceeding to data collection.")

    # If data was not loaded, proceed with data collection
    print(f"Collecting data from environment: {env_name}")
    # Try to make the environment, prioritizing 'rgb_array' for image collection,
    # as this function is designed to feed an image-based preprocessing pipeline.
    try:
        env = gym.make(env_name, render_mode='rgb_array')
        print(
            f"Successfully created env '{env_name}' with render_mode='rgb_array'.")
    except Exception as e_rgb:
        print(
            f"Failed to create env '{env_name}' with render_mode='rgb_array': {e_rgb}. Trying with render_mode=None...")
        try:
            env = gym.make(env_name, render_mode=None)
            print(
                f"Successfully created env '{env_name}' with render_mode=None.")
        except Exception as e_none:
            print(
                f"Failed to create env '{env_name}' with render_mode=None: {e_none}. Trying without render_mode arg...")
            env = gym.make(env_name)  # Fallback
            print(
                f"Successfully created env '{env_name}' with default render_mode ('{env.render_mode if hasattr(env, 'render_mode') else 'unknown'}').")

    all_episodes_raw_data = []  # Stores list of lists of (s,a,r,s') tuples

    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize(image_size),
        T.ToTensor()
    ])

    for episode_idx in range(num_episodes):
        current_state_img, info = env.reset()
        # Standardize image check after reset
        # If the observation is not a uint8 numpy array, it might be a float array (like CartPole)
        # or something else. Try to render if the env is in 'rgb_array' mode.
        initial_obs_is_uint8_image = isinstance(
            current_state_img, np.ndarray) and current_state_img.dtype == np.uint8

        if not initial_obs_is_uint8_image:
            # If initial obs is not a uint8 image, and env is in 'rgb_array' mode, try to render.
            if env.render_mode == 'rgb_array':
                print(
                    f"Warning: Initial observation for episode {episode_idx+1} is not uint8. Attempting render due to env.render_mode='rgb_array'. Original obs type: {type(current_state_img)}, dtype: {current_state_img.dtype if hasattr(current_state_img, 'dtype') else 'N/A'}")
                current_state_img = env.render()  # This should now provide an image
                if not (isinstance(current_state_img, np.ndarray) and current_state_img.dtype == np.uint8):
                    print(
                        f"Error: env.render() in rgb_array mode did not return a uint8 numpy array for episode {episode_idx+1}. State: {current_state_img}. Skipping episode.")
                    continue
            else:
                # If not uint8 and not in rgb_array mode, we can't process it as an image.
                print(
                    f"Warning: Initial observation for episode {episode_idx+1} is not a uint8 image and env.render_mode is '{env.render_mode}'. Cannot process as image. Skipping episode. Observation: {current_state_img}")
                continue

        # After potential rendering, re-check if we have a valid image structure (at least 2D)
        if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2):
            print(
                f"Skipping episode {episode_idx+1} due to unsuitable initial state after potential render. State: {current_state_img}")
            continue

        episode_transitions = []
        terminated = False
        truncated = False
        step_count = 0
        cumulative_reward_episode = 0.0  # Initialize cumulative reward

        while not (terminated or truncated) and step_count < max_steps_per_episode:
            action = env.action_space.sample()
            next_state_img, reward, terminated, truncated, info = env.step(
                action)

            # Standardize image check for next_state
            next_obs_is_uint8_image = isinstance(
                next_state_img, np.ndarray) and next_state_img.dtype == np.uint8

            if not next_obs_is_uint8_image:
                if env.render_mode == 'rgb_array':
                    # print(f"Debug: Next state obs for ep {episode_idx+1}, step {step_count+1} is not uint8. Attempting render. Orig type: {type(next_state_img)}")
                    next_state_img = env.render()
                    if not (isinstance(next_state_img, np.ndarray) and next_state_img.dtype == np.uint8):
                        print(
                            f"Error: env.render() for next_state in rgb_array mode did not return a uint8 numpy array for ep {episode_idx+1}, step {step_count+1}. State: {next_state_img}. Skipping step.")
                        step_count += 1
                        if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2):
                            break
                        else:
                            continue
                else:
                    print(
                        f"Warning: Next state obs for ep {episode_idx+1}, step {step_count+1} is not uint8 and env.render_mode is '{env.render_mode}'. Cannot process as image. Skipping step. Obs: {next_state_img}")
                    step_count += 1
                    if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2):
                        break
                    else:
                        continue

            # Ensure both current and next states are valid image-like arrays after potential rendering
            if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2 and
                    isinstance(next_state_img, np.ndarray) and next_state_img.ndim >= 2):
                print(f"Warning: Skipping step in episode {episode_idx+1}, step {step_count+1} due to unsuitable state dimensions after potential render. Current shape: {current_state_img.shape if hasattr(current_state_img, 'shape') else 'N/A'}, Next shape: {next_state_img.shape if hasattr(next_state_img, 'shape') else 'N/A'}")
                # Try to recover with next_state_img if it's valid
                current_state_img = next_state_img
                step_count += 1
                if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2):
                    break  # If new current_state_img is also bad
                else:
                    # Try next step with the (potentially problematic) next_state_img as current
                    continue

            episode_transitions.append(
                (current_state_img, action, reward, next_state_img))
            current_state_img = next_state_img
            step_count += 1

        if episode_transitions:
            all_episodes_raw_data.append(episode_transitions)
        print(f"Episode {episode_idx+1}/{num_episodes} finished after {step_count} steps. Collected {len(episode_transitions)} transitions.")

    env.close()

    if not all_episodes_raw_data:
        print("No data collected. Returning empty datasets.")
        empty_dataset = ExperienceDataset([], [], [], [], transform=preprocess)
        return empty_dataset, empty_dataset

    random.shuffle(all_episodes_raw_data)

    num_total_episodes = len(all_episodes_raw_data)
    # Index for end of training set
    split_idx = int((1.0 - validation_split_ratio) * num_total_episodes)

    train_episodes_list = all_episodes_raw_data[:split_idx]
    val_episodes_list = all_episodes_raw_data[split_idx:]

    print(f"Total episodes collected: {num_total_episodes}")
    print(
        f"Splitting into {len(train_episodes_list)} training episodes and {len(val_episodes_list)} validation episodes.")

    def create_dataset_from_episode_list(episode_list, transform_fn):
        flat_states, flat_actions, flat_rewards, flat_next_states = [], [], [], []
        for episode_data in episode_list:
            for s, a, r, ns in episode_data:
                flat_states.append(s)
                flat_actions.append(a)
                flat_rewards.append(r)
                flat_next_states.append(ns)

        # If flat_states is empty, ExperienceDataset will handle it (or should)
        return ExperienceDataset(flat_states, flat_actions, flat_rewards, flat_next_states, transform=transform_fn)

    train_dataset = create_dataset_from_episode_list(
        train_episodes_list, preprocess)
    validation_dataset = create_dataset_from_episode_list(
        val_episodes_list, preprocess)

    print(f"Training dataset: {len(train_dataset)} transitions.")
    print(f"Validation dataset: {len(validation_dataset)} transitions.")

    # Save the collected dataset if new data was collected
    if not data_loaded_successfully:
        if all_episodes_raw_data: # Check if new data was actually collected
            metadata_to_save = {
                'environment_name': env_name,
                'num_episodes_collected': config['num_episodes_data_collection'], # Or len(all_episodes_raw_data)
                'image_size': image_size,
                'max_steps_per_episode': max_steps_per_episode,
                'validation_split_ratio': validation_split_ratio,
                'num_train_transitions': len(train_dataset),
                'num_val_transitions': len(validation_dataset),
                'collection_method': 'random'
            }

            actual_save_path = os.path.join(dataset_dir, save_file_name_with_ext)

            def get_np_array(data_list, is_object_array):
                if not data_list:
                    return np.array([], dtype=object) if is_object_array else np.array([])
                return np.asarray(data_list, dtype=object) if is_object_array else np.asarray(data_list)

            train_states_np = get_np_array(train_dataset.states, True)
            train_actions_np = get_np_array(train_dataset.actions, False)
            train_rewards_np = get_np_array(train_dataset.rewards, False)
            train_next_states_np = get_np_array(train_dataset.next_states, True)

            val_states_np = get_np_array(validation_dataset.states, True)
            val_actions_np = get_np_array(validation_dataset.actions, False)
            val_rewards_np = get_np_array(validation_dataset.rewards, False)
            val_next_states_np = get_np_array(validation_dataset.next_states, True)

            data_to_save_npz = {
                'train_states': train_states_np,
                'train_actions': train_actions_np,
                'train_rewards': train_rewards_np,
                'train_next_states': train_next_states_np,
                'val_states': val_states_np,
                'val_actions': val_actions_np,
                'val_rewards': val_rewards_np,
                'val_next_states': val_next_states_np,
                'metadata': np.array([pickle.dumps(metadata_to_save)], dtype=object)
            }

            try:
                np.savez_compressed(actual_save_path, **data_to_save_npz)
                print(f"Dataset saved to {actual_save_path}")
            except Exception as e:
                print(f"Error saving dataset to {actual_save_path}: {e}")
        else:
            print("No new data was collected, so dataset will not be saved.")


    return train_dataset, validation_dataset


def collect_ppo_episodes(config, max_steps_per_episode, image_size, validation_split_ratio):
    env_name = config['environment_name']
    num_episodes = config['num_episodes_data_collection']

    dataset_dir = config['dataset_dir']
    load_dataset_filename_config = config['load_dataset_path']
    save_dataset_filename_config = config['dataset_filename']

    os.makedirs(dataset_dir, exist_ok=True)

    # Determine save filename with .npz extension
    save_file_name_with_ext = save_dataset_filename_config
    if not save_file_name_with_ext.lower().endswith('.npz'):
        save_file_name_with_ext = os.path.splitext(save_file_name_with_ext)[0] + '.npz'

    # Determine load filename with .npz extension (if load_dataset_filename is provided)
    load_file_name_with_ext = None
    if load_dataset_filename_config:
        load_file_name_with_ext = load_dataset_filename_config
        if not load_file_name_with_ext.lower().endswith('.npz'):
            load_file_name_with_ext = os.path.splitext(load_file_name_with_ext)[0] + '.npz'

    data_loaded_successfully = False
    if load_file_name_with_ext:
        dataset_path_to_load = os.path.join(dataset_dir, load_file_name_with_ext)
        if os.path.exists(dataset_path_to_load):
            print(f"Loading dataset from {dataset_path_to_load}...")
            try:
                loaded_data = np.load(dataset_path_to_load, allow_pickle=True)

                metadata_pickled_array = loaded_data['metadata']
                if metadata_pickled_array.ndim == 0 or metadata_pickled_array.size == 1:
                    metadata = pickle.loads(metadata_pickled_array.item())
                else:
                    raise ValueError("Metadata format error in .npz file.")

                loaded_env_name = metadata.get('environment_name')
                if loaded_env_name != env_name:
                    print(f"Error: Mismatch between loaded dataset environment ('{loaded_env_name}') and config environment ('{env_name}').")
                    raise ValueError("Environment mismatch in loaded dataset.")

                if metadata.get('collection_method') != 'ppo':
                    print(f"Warning: Loaded dataset from {dataset_path_to_load} was collected using '{metadata.get('collection_method', 'unknown')}' not 'ppo'. Proceeding to collect new data with PPO.")
                else:
                    preprocess = T.Compose([
                        T.ToPILImage(),
                        T.Resize(image_size),
                        T.ToTensor()
                    ])

                    def get_list_from_npz(npz_file, key):
                        if key not in npz_file or npz_file[key].size == 0:
                            return []
                        return list(npz_file[key])

                    train_states_loaded = get_list_from_npz(loaded_data, 'train_states')
                    train_actions_loaded = get_list_from_npz(loaded_data, 'train_actions')
                    train_rewards_loaded = get_list_from_npz(loaded_data, 'train_rewards')
                    train_next_states_loaded = get_list_from_npz(loaded_data, 'train_next_states')

                    val_states_loaded = get_list_from_npz(loaded_data, 'val_states')
                    val_actions_loaded = get_list_from_npz(loaded_data, 'val_actions')
                    val_rewards_loaded = get_list_from_npz(loaded_data, 'val_rewards')
                    val_next_states_loaded = get_list_from_npz(loaded_data, 'val_next_states')

                    loaded_train_dataset = ExperienceDataset(
                        train_states_loaded, train_actions_loaded, train_rewards_loaded, train_next_states_loaded,
                        transform=preprocess
                    )
                    loaded_val_dataset = ExperienceDataset(
                        val_states_loaded, val_actions_loaded, val_rewards_loaded, val_next_states_loaded,
                        transform=preprocess
                    )
                    print(f"Successfully loaded PPO-collected dataset for environment '{loaded_env_name}' ... from {dataset_path_to_load}.")
                    data_loaded_successfully = True
                    return loaded_train_dataset, loaded_val_dataset
            except Exception as e:
                print(f"Error loading dataset from {dataset_path_to_load}: {e}. Proceeding to PPO data collection.")
        else:
            print(f"Warning: Dataset {dataset_path_to_load} not found. Proceeding to PPO data collection.")
    else:
        print("`load_dataset_path` (or its .npz version) is effectively empty or not found. Proceeding to PPO data collection.")

    print(f"Collecting data from environment: {env_name} using PPO agent.")
    try:
        env = gym.make(env_name, render_mode='rgb_array')
        print(f"Successfully created env '{env_name}' with render_mode='rgb_array' for PPO collection.")
    except Exception as e_rgb:
        print(f"Failed to create env '{env_name}' with render_mode='rgb_array': {e_rgb}. Trying with render_mode=None...")
        try:
            env = gym.make(env_name, render_mode=None)
            print(f"Successfully created env '{env_name}' with render_mode=None for PPO collection.")
        except Exception as e_none:
            print(f"Failed to create env '{env_name}' with render_mode=None: {e_none}. Trying without render_mode arg...")
            env = gym.make(env_name)
            print(f"Successfully created env '{env_name}' with default render_mode ('{env.render_mode if hasattr(env, 'render_mode') else 'unknown'}') for PPO collection.")

    # PPO Agent Setup
    ppo_specific_config = config.get('ppo_agent', {})
    if not ppo_specific_config or not ppo_specific_config.get('enabled', False):
        print("PPO agent configuration is missing or disabled in config. Cannot collect PPO episodes.")
        # Return empty datasets or raise an error
        empty_dataset = ExperienceDataset([], [], [], [], transform=T.Compose([T.ToPILImage(),T.Resize(image_size),T.ToTensor()]))
        return empty_dataset, empty_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device  # For Apple Silicon Macs
    print(f"Using device: {device} for PPO agent.")

    # Create a temporary env for PPO training if needed, or use the main `env`
    # It's better to pass the same env instance to ensure compatibility of obs/action spaces
    ppo_agent = create_ppo_agent(env, ppo_specific_config, device=device)
    train_ppo_agent(ppo_agent, ppo_specific_config, task_name="Initial PPO Training for Data Collection")

    additional_noise = ppo_specific_config.get('additional_log_std_noise', 0.0)
    if additional_noise != 0.0: # Only proceed if noise is non-zero
        if hasattr(ppo_agent.policy, 'log_std') and isinstance(ppo_agent.policy.log_std, torch.Tensor):
            current_log_std = ppo_agent.policy.log_std.data
            noise_tensor = torch.tensor(additional_noise, device=current_log_std.device, dtype=current_log_std.dtype)
            ppo_agent.policy.log_std.data += noise_tensor
            print(f"Adjusted PPO policy log_std by {additional_noise:.4f}")
        elif hasattr(ppo_agent.policy, 'action_dist') and hasattr(ppo_agent.policy.action_dist, 'log_std_param'): # For SquashedGaussian
            current_log_std_param = ppo_agent.policy.action_dist.log_std_param
            if isinstance(current_log_std_param, torch.Tensor):
                noise_tensor = torch.tensor(additional_noise, device=current_log_std_param.device, dtype=current_log_std_param.dtype)
                # For nn.Parameters, modification should be in-place on .data or via an optimizer step if it were training
                current_log_std_param.data += noise_tensor
                print(f"Adjusted PPO policy action_dist.log_std_param by {additional_noise:.4f}")
            else:
                print(f"Warning: PPO policy action_dist.log_std_param found but is not a Tensor. Type: {type(ppo_agent.policy.action_dist.log_std_param)}. Skipping noise addition.")
        else:
            print(f"Warning: PPO policy does not have a 'log_std' Tensor or 'action_dist.log_std_param' Tensor. Skipping noise addition to log_std.")

    # After training, the env used by PPO (which is `env` wrapped in DummyVecEnv) should still be usable
    # as SB3 usually doesn't close the original envs passed to DummyVecEnv unless DummyVecEnv.close() is called,
    # which happens if ppo_agent.env.close() is called. `learn()` does not close it.

    all_episodes_raw_data = []
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize(image_size),
        T.ToTensor()
    ])

    for episode_idx in range(num_episodes):
        current_state_img, info = env.reset()
        initial_obs_is_uint8_image = isinstance(current_state_img, np.ndarray) and current_state_img.dtype == np.uint8

        if not initial_obs_is_uint8_image:
            if env.render_mode == 'rgb_array':
                current_state_img = env.render()
                if not (isinstance(current_state_img, np.ndarray) and current_state_img.dtype == np.uint8):
                    print(f"Error: env.render() did not return a uint8 numpy array for PPO ep {episode_idx+1}. Skipping.")
                    continue
            else:
                print(f"Warning: Initial PPO obs for ep {episode_idx+1} not uint8 and env not in 'rgb_array' mode. Skipping. Obs: {current_state_img}")
                continue

        if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2):
            print(f"Skipping PPO episode {episode_idx+1} due to unsuitable initial state. State: {current_state_img}")
            continue

        episode_transitions = []
        terminated = False
        truncated = False
        step_count = 0
        cumulative_reward_episode = 0.0  # Initialize cumulative reward

        while not (terminated or truncated) and step_count < max_steps_per_episode:
            # Action selection by PPO agent
            # current_state_img is HWC, uint8 numpy array. SB3 CnnPolicy expects this.
            action, _ = ppo_agent.predict(current_state_img, deterministic=True) # Use deterministic for collection consistency

            next_state_img, reward, terminated, truncated, info = env.step(action)
            cumulative_reward_episode += reward  # Add reward to cumulative sum
            next_obs_is_uint8_image = isinstance(next_state_img, np.ndarray) and next_state_img.dtype == np.uint8

            if not next_obs_is_uint8_image:
                if env.render_mode == 'rgb_array':
                    next_state_img = env.render()
                    if not (isinstance(next_state_img, np.ndarray) and next_state_img.dtype == np.uint8):
                        print(f"Error: env.render() for next_state (PPO) did not return uint8 array for ep {episode_idx+1}, step {step_count+1}. Skipping step.")
                        step_count += 1
                        if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2): break
                        else: continue
                else:
                    print(f"Warning: Next PPO obs for ep {episode_idx+1}, step {step_count+1} not uint8 and env not 'rgb_array'. Skipping step. Obs: {next_state_img}")
                    step_count += 1
                    if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2): break
                    else: continue

            if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2 and
                    isinstance(next_state_img, np.ndarray) and next_state_img.ndim >= 2):
                print(f"Warning: Skipping PPO step in ep {episode_idx+1}, step {step_count+1} due to unsuitable state dims. Current: {current_state_img.shape if hasattr(current_state_img, 'shape') else 'N/A'}, Next: {next_state_img.shape if hasattr(next_state_img, 'shape') else 'N/A'}")
                current_state_img = next_state_img
                step_count += 1
                if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2): break
                else: continue

            episode_transitions.append((current_state_img, action.item() if isinstance(action, np.ndarray) and env.action_space.shape == () else action, reward, next_state_img))
            current_state_img = next_state_img
            step_count += 1

        if episode_transitions:
            all_episodes_raw_data.append(episode_transitions)
        print(f"PPO Episode {episode_idx+1}/{num_episodes} finished after {step_count} steps. Cumulative Reward: {cumulative_reward_episode:.2f}. Collected {len(episode_transitions)} transitions.")

    env.close() # Close the environment used for PPO collection

    if not all_episodes_raw_data:
        print("No data collected with PPO. Returning empty datasets.")
        empty_dataset = ExperienceDataset([], [], [], [], transform=preprocess)
        return empty_dataset, empty_dataset

    random.shuffle(all_episodes_raw_data)
    num_total_episodes = len(all_episodes_raw_data)
    split_idx = int((1.0 - validation_split_ratio) * num_total_episodes)
    train_episodes_list = all_episodes_raw_data[:split_idx]
    val_episodes_list = all_episodes_raw_data[split_idx:]

    print(f"Total PPO episodes collected: {num_total_episodes}")
    print(f"Splitting into {len(train_episodes_list)} training episodes and {len(val_episodes_list)} validation episodes.")

    def create_dataset_from_episode_list(episode_list, transform_fn):
        flat_states, flat_actions, flat_rewards, flat_next_states = [], [], [], []
        for episode_data in episode_list:
            for s, a, r, ns in episode_data:
                flat_states.append(s)
                flat_actions.append(a)
                flat_rewards.append(r)
                flat_next_states.append(ns)
        return ExperienceDataset(flat_states, flat_actions, flat_rewards, flat_next_states, transform=transform_fn)

    train_dataset = create_dataset_from_episode_list(train_episodes_list, preprocess)
    validation_dataset = create_dataset_from_episode_list(val_episodes_list, preprocess)

    print(f"PPO Training dataset: {len(train_dataset)} transitions.")
    print(f"PPO Validation dataset: {len(validation_dataset)} transitions.")

    if not data_loaded_successfully:
        if all_episodes_raw_data:
            metadata_to_save = {
                'environment_name': env_name,
                'num_episodes_collected': config['num_episodes_data_collection'],
                'image_size': image_size,
                'max_steps_per_episode': max_steps_per_episode,
                'validation_split_ratio': validation_split_ratio,
                'num_train_transitions': len(train_dataset),
                'num_val_transitions': len(validation_dataset),
                'collection_method': 'ppo',
                'ppo_config_params': ppo_specific_config
            }

            actual_save_path = os.path.join(dataset_dir, save_file_name_with_ext)

            def get_np_array(data_list, is_object_array):
                if not data_list:
                    return np.array([], dtype=object) if is_object_array else np.array([])
                return np.asarray(data_list, dtype=object) if is_object_array else np.asarray(data_list)

            train_states_np = get_np_array(train_dataset.states, True)
            train_actions_np = get_np_array(train_dataset.actions, False)
            train_rewards_np = get_np_array(train_dataset.rewards, False)
            train_next_states_np = get_np_array(train_dataset.next_states, True)

            val_states_np = get_np_array(validation_dataset.states, True)
            val_actions_np = get_np_array(validation_dataset.actions, False)
            val_rewards_np = get_np_array(validation_dataset.rewards, False)
            val_next_states_np = get_np_array(validation_dataset.next_states, True)

            data_to_save_npz = {
                'train_states': train_states_np,
                'train_actions': train_actions_np,
                'train_rewards': train_rewards_np,
                'train_next_states': train_next_states_np,
                'val_states': val_states_np,
                'val_actions': val_actions_np,
                'val_rewards': val_rewards_np,
                'val_next_states': val_next_states_np,
                'metadata': np.array([pickle.dumps(metadata_to_save)], dtype=object)
            }

            try:
                np.savez_compressed(actual_save_path, **data_to_save_npz)
                print(f"PPO collected dataset saved to {actual_save_path}")
            except Exception as e:
                print(f"Error saving PPO dataset to {actual_save_path}: {e}")
        else:
            print("No new PPO data was collected, so dataset will not be saved.")

    return train_dataset, validation_dataset


if __name__ == '__main__':
    # Example usage:
    # Ensure you have a display server if using environments like CarRacing-v2 locally without headless mode.
    # For servers, use Xvfb: Xvfb :1 -screen 0 1024x768x24 &
    # export DISPLAY=:1

    print(f"Testing data collection with a sample environment...")
    try:
        # Attempt to use a known pixel-based environment
        try:
            # Using CartPole-v1 and rendering it as an image substitute for Atari envs
            gym.make("CartPole-v1", render_mode='rgb_array') # Ensure rgb_array for rendering
            test_env_name = "CartPole-v1"
            print("Using CartPole-v1 (rendered) for testing data collection.")
        except Exception as e: # Catch any exception during env creation
            print(f"Could not create CartPole-v1 for testing: {e}. Skipping data_utils.py example run.")
            test_env_name = None


        if test_env_name:
            shared_dataset_filename = f"test_data_utils_{test_env_name.replace('/', '_')}.npz" # Unique name for test
            dataset_test_dir = "datasets_test/" # Use a dedicated test dataset directory
            os.makedirs(dataset_test_dir, exist_ok=True) # Ensure it exists

            dummy_config_collect = {
                'environment_name': test_env_name,
                'num_episodes_data_collection': 5,
                'dataset_dir': dataset_test_dir,
                'dataset_filename': shared_dataset_filename, # For saving
                'load_dataset_path': "", # Don't load for collection test
                'ppo_agent': { 'enabled': False }
            }

            print("\n--- Test Case 1: Collect and Save (NPZ) ---")
            train_d, val_d = collect_random_episodes(
                config=dummy_config_collect,
                max_steps_per_episode=50,
                image_size=(64, 64),
                validation_split_ratio=0.4
            )

            print(f"\n--- Training Dataset (Size: {len(train_d)}) ---")
            if len(train_d) > 0:
                train_dataloader = DataLoader(train_d, batch_size=4, shuffle=True)
                s_batch, a_batch, r_batch, s_next_batch = next(iter(train_dataloader))
                print(f"Training Sample batch shapes: States {s_batch.shape}, Actions {a_batch.shape}, Rewards {r_batch.shape}, Next States {s_next_batch.shape}")
            else:
                print("Training dataset is empty.")

            print(f"\n--- Validation Dataset (Size: {len(val_d)}) ---")
            if len(val_d) > 0:
                val_dataloader = DataLoader(val_d, batch_size=4, shuffle=False)
                s_val_batch, a_val_batch, r_val_batch, s_next_val_batch = next(iter(val_dataloader))
                print(f"Validation Sample batch shapes: States {s_val_batch.shape}, Actions {a_val_batch.shape}, Rewards {r_val_batch.shape}, Next States {s_next_val_batch.shape}")
            else:
                print("Validation dataset is empty.")

            dummy_config_load = {
                'environment_name': test_env_name,
                'num_episodes_data_collection': 0,
                'dataset_dir': dataset_test_dir,
                'load_dataset_path': shared_dataset_filename, # For loading
                'dataset_filename': "", # Not saving
                'ppo_agent': { 'enabled': False }
            }

            expected_file_path = os.path.join(dataset_test_dir, shared_dataset_filename)
            if os.path.exists(expected_file_path):
                print("\n--- Test Case 2: Load Saved Dataset (NPZ) ---")
                train_d_loaded, val_d_loaded = collect_random_episodes(
                    config=dummy_config_load,
                    max_steps_per_episode=50,
                    image_size=(64, 64),
                    validation_split_ratio=0.4
                )

                print(f"\n--- Loaded Training Dataset (Size: {len(train_d_loaded)}) ---")
                if len(train_d_loaded) > 0:
                    # Basic check: compare sizes with originally collected data
                    assert len(train_d_loaded) == len(train_d), "Loaded train dataset size mismatch!"
                    print("Loaded training dataset size matches original.")
                else:
                    # If original train_d was also empty, this is fine.
                    assert len(train_d) == 0, "Original training dataset was not empty but loaded one is."
                    print("Loaded training dataset is empty (matches original).")

                print(f"\n--- Loaded Validation Dataset (Size: {len(val_d_loaded)}) ---")
                if len(val_d_loaded) > 0:
                    assert len(val_d_loaded) == len(val_d), "Loaded validation dataset size mismatch!"
                    print("Loaded validation dataset size matches original.")
                else:
                    # If original val_d was also empty, this is fine.
                    assert len(val_d) == 0, "Original validation dataset was not empty but loaded one is."
                    print("Loaded validation dataset is empty (matches original).")

                # Add asserts to check dataset lengths more directly
                if len(train_d) > 0 and len(train_d_loaded) > 0: # Redundant with above but good for clarity
                    assert len(train_d) == len(train_d_loaded), "Loaded train dataset size mismatch!"
                    print("Loaded training dataset size matches original.")
                if len(val_d) > 0 and len(val_d_loaded) > 0: # Redundant
                    assert len(val_d) == len(val_d_loaded), "Loaded validation dataset size mismatch!"
                    print("Loaded validation dataset size matches original.")

            else:
                print(f"\n--- Test Case 2: Skipped --- Dataset file {expected_file_path} not found. Test Case 1 might have failed to save.")

    except ImportError as e:
        print(f"Import error during example run: {e}")
    except Exception as e:
        print(f"An error occurred during the example run: {e}")
        import traceback
        traceback.print_exc()
