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
        # Ensure states, actions, rewards, next_states are numpy arrays
        self.states = states # Should be np.array
        self.actions = actions # Should be np.array
        self.rewards = rewards # Should be np.array
        self.next_states = next_states # Should be np.array
        self.transform = transform

    def __len__(self):
        # All arrays should have the same first dimension (number of samples)
        if isinstance(self.states, np.ndarray):
            return self.states.shape[0]
        else: # Should not happen if constructor is used correctly
            return 0


    def __getitem__(self, idx):
        state = self.states[idx] # Accessing NumPy array
        action = self.actions[idx] # Accessing NumPy array
        reward = self.rewards[idx] # Accessing NumPy array
        next_state = self.next_states[idx] # Accessing NumPy array

        # State and next_state are expected to be images (e.g., HWC numpy arrays)
        # The transform will handle conversion to PILImage and then ToTensor
        if self.transform:
            state = self.transform(state)
            next_state = self.transform(next_state)

        # Convert action to tensor.
        # Actions can be discrete (scalar or int array) or continuous (float array).
        # Rewards are typically scalar floats.
        action_tensor = torch.tensor(action, dtype=torch.float32) # Adjust dtype as needed, esp for discrete
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        return state, action_tensor, reward_tensor, next_state


def collect_random_episodes(config, max_steps_per_episode, image_size, validation_split_ratio):
    env_name = config['environment_name']
    num_episodes = config['num_episodes_data_collection']

    # New configuration keys
    dataset_dir = config['dataset_dir']
    load_dataset_filename = config['load_dataset_path'] # Filename or path relative to dataset_dir
    save_dataset_filename = config['dataset_filename'] # Filename for saving new datasets

    os.makedirs(dataset_dir, exist_ok=True)

    data_loaded_successfully = False
    # Define preprocess transform here, as it's needed for both loading and new collection
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize(image_size),
        T.ToTensor()
    ])

    if load_dataset_filename: # If not an empty string, attempt to load
        load_base_name = load_dataset_filename
        for ext in ['.pkl', '.npz']: # Remove known extensions to get a clean base name
            if load_base_name.endswith(ext):
                load_base_name = load_base_name[:-len(ext)]
                break

        meta_path = os.path.join(dataset_dir, f"{load_base_name}_meta.pkl")
        train_npz_path = os.path.join(dataset_dir, f"{load_base_name}_train.npz")
        val_npz_path = os.path.join(dataset_dir, f"{load_base_name}_val.npz")

        if os.path.exists(meta_path):
            print(f"Loading metadata from {meta_path}...")
            try:
                with open(meta_path, 'rb') as f:
                    metadata = pickle.load(f)

                loaded_env_name = metadata.get('environment_name')
                if loaded_env_name != env_name:
                    print(f"Error: Mismatch between loaded dataset environment ('{loaded_env_name}') and config environment ('{env_name}').")
                    raise ValueError("Environment mismatch in loaded dataset.")

                print(f"Loading training data from {train_npz_path}...")
                train_data_npz = np.load(train_npz_path)
                loaded_train_states = train_data_npz['states']
                loaded_train_actions = train_data_npz['actions']
                loaded_train_rewards = train_data_npz['rewards']
                loaded_train_next_states = train_data_npz['next_states']
                train_dataset = ExperienceDataset(loaded_train_states, loaded_train_actions, loaded_train_rewards, loaded_train_next_states, transform=preprocess)

                loaded_val_dataset = None
                if os.path.exists(val_npz_path):
                    print(f"Loading validation data from {val_npz_path}...")
                    val_data_npz = np.load(val_npz_path)
                    if 'states' in val_data_npz and val_data_npz['states'].shape[0] > 0: # Check if val data is not empty
                        loaded_val_states = val_data_npz['states']
                        loaded_val_actions = val_data_npz['actions']
                        loaded_val_rewards = val_data_npz['rewards']
                        loaded_val_next_states = val_data_npz['next_states']
                        validation_dataset = ExperienceDataset(loaded_val_states, loaded_val_actions, loaded_val_rewards, loaded_val_next_states, transform=preprocess)
                    else:
                        print("Validation data file found but appears empty. Creating an empty validation dataset.")
                        validation_dataset = ExperienceDataset(np.array([]), np.array([]), np.array([]), np.array([]), transform=preprocess)
                else:
                    print(f"Validation data file {val_npz_path} not found. Creating an empty validation dataset.")
                    validation_dataset = ExperienceDataset(np.array([]), np.array([]), np.array([]), np.array([]), transform=preprocess)

                print(f"Successfully loaded dataset for environment '{loaded_env_name}' with {metadata.get('num_episodes_collected', 'N/A')} episodes (from metadata).")
                data_loaded_successfully = True
                return train_dataset, validation_dataset
            except Exception as e:
                print(f"Error loading dataset parts (meta: {meta_path}, train: {train_npz_path}, val: {val_npz_path}): {e}. Proceeding to data collection.")
        else:
            print(f"Warning: Metadata file {meta_path} not found. Proceeding to data collection.")
    else:
        print("`load_dataset_path` is empty. Proceeding to data collection.")

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
                f"Successfully created env '{env_name}' with render_mode ('{env.render_mode if hasattr(env, 'render_mode') else 'unknown'}').")

    # Determine observation shape from a sample observation
    # This requires resetting the env and getting one observation.
    _obs_sample, _ = env.reset()
    if not (isinstance(_obs_sample, np.ndarray) and _obs_sample.dtype == np.uint8):
        if env.render_mode == 'rgb_array':
            _obs_sample = env.render()
    if not (isinstance(_obs_sample, np.ndarray) and _obs_sample.ndim >=2): # Must be image like
         env.close()
         raise ValueError(f"Initial observation from environment {env_name} is not suitable for image processing. Shape: {_obs_sample.shape if hasattr(_obs_sample, 'shape') else 'N/A'}, dtype: {_obs_sample.dtype if hasattr(_obs_sample, 'dtype') else 'N/A'}")
    obs_shape = _obs_sample.shape

    # Determine action shape and dtype
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Discrete):
        action_shape = () # Scalar action
        action_dtype = np.int_ # Default int type for discrete actions
    elif isinstance(action_space, gym.spaces.Box):
        action_shape = action_space.shape
        action_dtype = action_space.dtype
    else:
        env.close()
        raise TypeError(f"Unsupported action space type: {type(action_space)}")

    max_total_transitions = num_episodes * max_steps_per_episode

    # Pre-allocate NumPy arrays
    all_states = np.zeros((max_total_transitions, *obs_shape), dtype=np.uint8)
    all_next_states = np.zeros((max_total_transitions, *obs_shape), dtype=np.uint8)
    # For action_shape that might be (), np.zeros needs it as (max_total_transitions,)
    _act_shape_for_zeros = (max_total_transitions,) + action_shape if action_shape else (max_total_transitions,)
    all_actions = np.zeros(_act_shape_for_zeros, dtype=action_dtype)
    all_rewards = np.zeros(max_total_transitions, dtype=np.float32)
    all_episode_done_flags = np.zeros(max_total_transitions, dtype=bool) # True if transition was the last in an episode

    transition_counter = 0
    actual_episodes_collected = 0

    # preprocess is already defined above

    for episode_idx in range(num_episodes):
        current_state_img, info = env.reset()
        initial_obs_is_uint8_image = isinstance(current_state_img, np.ndarray) and current_state_img.dtype == np.uint8

        if not initial_obs_is_uint8_image:
            if env.render_mode == 'rgb_array':
                current_state_img = env.render()
                if not (isinstance(current_state_img, np.ndarray) and current_state_img.dtype == np.uint8):
                    print(f"Error: env.render() in rgb_array mode did not return a uint8 numpy array for episode {episode_idx+1}. Skipping episode.")
                    continue
            else:
                print(f"Warning: Initial observation for episode {episode_idx+1} is not a uint8 image and env.render_mode is '{env.render_mode}'. Skipping episode.")
                continue

        if not (isinstance(current_state_img, np.ndarray) and current_state_img.ndim >= 2 and current_state_img.shape == obs_shape):
            print(f"Skipping episode {episode_idx+1} due to unsuitable initial state shape. Expected {obs_shape}, got {current_state_img.shape if hasattr(current_state_img, 'shape') else 'N/A'}")
            continue

        terminated = False
        truncated = False
        step_count = 0

        episode_has_transitions = False # Flag to track if any transitions were added for this episode

        while not (terminated or truncated) and step_count < max_steps_per_episode:
            if transition_counter >= max_total_transitions:
                print("Warning: Reached max_total_transitions. Stopping data collection early.")
                # This break will exit the inner while loop for the current episode.
                # The outer for loop for episodes will also need to be broken.
                break

            action = env.action_space.sample()
            next_state_img, reward, terminated, truncated, info = env.step(action)
            next_obs_is_uint8_image = isinstance(next_state_img, np.ndarray) and next_state_img.dtype == np.uint8

            if not next_obs_is_uint8_image:
                if env.render_mode == 'rgb_array':
                    next_state_img = env.render()
                    if not (isinstance(next_state_img, np.ndarray) and next_state_img.dtype == np.uint8):
                        print(f"Error: env.render() for next_state did not return uint8 array for ep {episode_idx+1}, step {step_count+1}. Skipping step.")
                        # current_state_img remains the same, try next action from this state
                        step_count += 1
                        continue

            # Ensure states are valid and have the expected shape
            if not (isinstance(current_state_img, np.ndarray) and current_state_img.shape == obs_shape and
                    isinstance(next_state_img, np.ndarray) and next_state_img.shape == obs_shape):
                print(f"Warning: Skipping step in episode {episode_idx+1}, step {step_count+1} due to unsuitable state dimensions. Current: {current_state_img.shape if hasattr(current_state_img, 'shape') else 'N/A'}, Next: {next_state_img.shape if hasattr(next_state_img, 'shape') else 'N/A'}. Expected: {obs_shape}")
                current_state_img = next_state_img # Try to recover with next_state_img
                step_count += 1
                if not (isinstance(current_state_img, np.ndarray) and current_state_img.shape == obs_shape):
                    print(f"Recovery failed, current_state_img is unsuitable. Breaking episode.")
                    break # Break from while loop (current episode)
                else:
                    continue # Try next step with the recovered current_state_img

            all_states[transition_counter] = current_state_img
            all_actions[transition_counter] = action
            all_rewards[transition_counter] = reward
            all_next_states[transition_counter] = next_state_img
            all_episode_done_flags[transition_counter] = terminated or truncated

            current_state_img = next_state_img
            step_count += 1
            transition_counter += 1
            episode_has_transitions = True

        if episode_has_transitions: # Only increment if the episode contributed transitions
            actual_episodes_collected += 1

        print(f"Episode {episode_idx+1}/{num_episodes} finished after {step_count} steps. Total transitions collected: {transition_counter}")

        if transition_counter >= max_total_transitions: # Check after episode finishes
            print("Reached max_total_transitions. Stopping data collection.")
            break # Break from outer for loop (episodes)

    env.close()

    if transition_counter == 0:
        print("No data collected. Returning empty datasets.")
        # Ensure empty arrays have a dimension for shape[0] to work in ExperienceDataset
        empty_states_arr = np.zeros((0, *obs_shape), dtype=np.uint8) if obs_shape else np.array([])
        _empty_act_shape = (0,) + action_shape if action_shape else (0,)
        empty_actions_arr = np.zeros(_empty_act_shape, dtype=action_dtype) if action_dtype else np.array([])

        empty_train_dataset = ExperienceDataset(empty_states_arr, empty_actions_arr, np.array([]), empty_states_arr, transform=preprocess)
        empty_val_dataset = ExperienceDataset(empty_states_arr, empty_actions_arr, np.array([]), empty_states_arr, transform=preprocess)
        return empty_train_dataset, empty_val_dataset

    # Trim arrays to actual number of collected transitions
    final_states = all_states[:transition_counter]
    final_actions = all_actions[:transition_counter]
    final_rewards = all_rewards[:transition_counter]
    final_next_states = all_next_states[:transition_counter]
    final_done_flags = all_episode_done_flags[:transition_counter]

    # --- Episode-preserving shuffle and split ---
    episode_boundaries = np.where(final_done_flags)[0] + 1 # Indices *after* episode ends
    episode_starts = np.concatenate(([0], episode_boundaries[:-1])) # Add 0 for the start of the first episode

    episode_segments = [] # List of (start_idx, end_idx) tuples for each episode
    for start, end in zip(episode_starts, episode_boundaries):
        if end > start: # Ensure episode has at least one transition
             episode_segments.append((start, end))

    if not episode_segments: # Should be caught by transition_counter == 0, but as a safeguard
        print("No valid episode segments found after processing. Returning empty datasets.")
        empty_states_arr = np.zeros((0, *obs_shape), dtype=np.uint8)
        _empty_act_shape = (0,) + action_shape if action_shape else (0,)
        empty_actions_arr = np.zeros(_empty_act_shape, dtype=action_dtype)
        empty_train_dataset = ExperienceDataset(empty_states_arr, empty_actions_arr, np.array([]), empty_states_arr, transform=preprocess)
        empty_val_dataset = ExperienceDataset(empty_states_arr, empty_actions_arr, np.array([]), empty_states_arr, transform=preprocess)
        return empty_train_dataset, empty_val_dataset

    random.shuffle(episode_segments)

    num_total_actual_episodes = len(episode_segments)
    split_idx_episodes = int((1.0 - validation_split_ratio) * num_total_actual_episodes)

    train_episode_segments = episode_segments[:split_idx_episodes]
    val_episode_segments = episode_segments[split_idx_episodes:]

    print(f"Total actual episodes collected and processed: {num_total_actual_episodes}")
    print(f"Splitting into {len(train_episode_segments)} training episodes and {len(val_episode_segments)} validation episodes.")

    # Concatenate segments to form final train/val datasets
    # This can be memory intensive if we copy all data again.
    # A more efficient way is to create lists of indices for train/val and then use fancy indexing.

    train_indices = np.concatenate([np.arange(start, end) for start, end in train_episode_segments] if train_episode_segments else [[]]).astype(int)
    val_indices = np.concatenate([np.arange(start, end) for start, end in val_episode_segments] if val_episode_segments else [[]]).astype(int)

    if len(train_indices) > 0:
        train_s_np = final_states[train_indices]
        train_a_np = final_actions[train_indices]
        train_r_np = final_rewards[train_indices]
        train_ns_np = final_next_states[train_indices]
    else: # Create empty arrays with correct dimensions
        train_s_np = np.zeros((0, *obs_shape), dtype=np.uint8)
        _empty_act_shape_train = (0,) + action_shape if action_shape else (0,)
        train_a_np = np.zeros(_empty_act_shape_train, dtype=action_dtype)
        train_r_np = np.array([], dtype=np.float32)
        train_ns_np = np.zeros((0, *obs_shape), dtype=np.uint8)

    if len(val_indices) > 0:
        val_s_np = final_states[val_indices]
        val_a_np = final_actions[val_indices]
        val_r_np = final_rewards[val_indices]
        val_ns_np = final_next_states[val_indices]
    else: # Create empty arrays with correct dimensions
        val_s_np = np.zeros((0, *obs_shape), dtype=np.uint8)
        _empty_act_shape_val = (0,) + action_shape if action_shape else (0,)
        val_a_np = np.zeros(_empty_act_shape_val, dtype=action_dtype)
        val_r_np = np.array([], dtype=np.float32)
        val_ns_np = np.zeros((0, *obs_shape), dtype=np.uint8)

    print(f"Total transitions collected: {transition_counter}")
    # The create_dataset_from_episode_list function is no longer needed here,
    # as train_s_np, train_a_np etc. are already the final numpy arrays.

    train_dataset = ExperienceDataset(train_s_np, train_a_np, train_r_np, train_ns_np, transform=preprocess)
    validation_dataset = ExperienceDataset(val_s_np, val_a_np, val_r_np, val_ns_np, transform=preprocess)

    print(f"Training dataset: {len(train_dataset)} transitions.")
    print(f"Validation dataset: {len(validation_dataset)} transitions.")

    # Save the collected dataset if new data was collected
    if not data_loaded_successfully:
        if transition_counter > 0: # Check if new data was actually collected (transition_counter > 0)
            # num_episodes_collected in config is the target, actual_episodes_collected is what we got

            save_base_name = save_dataset_filename
            for ext in ['.pkl', '.npz']: # Remove known extensions
                if save_base_name.endswith(ext):
                    save_base_name = save_base_name[:-len(ext)]
                    break

            meta_save_path = os.path.join(dataset_dir, f"{save_base_name}_meta.pkl")
            train_npz_save_path = os.path.join(dataset_dir, f"{save_base_name}_train.npz")
            val_npz_save_path = os.path.join(dataset_dir, f"{save_base_name}_val.npz")

            metadata_to_save = {
                'environment_name': env_name,
                'num_episodes_collected': num_episodes, # Target num_episodes from config
                'actual_episodes_collected': actual_episodes_collected, # Actual number of episodes that yielded transitions
                'image_size': image_size,
                'max_steps_per_episode': max_steps_per_episode,
                'validation_split_ratio': validation_split_ratio,
                'num_train_transitions': len(train_dataset),
                'num_val_transitions': len(validation_dataset),
                'collection_method': 'random'
            }

            try:
                # Save metadata
                with open(meta_save_path, 'wb') as f:
                    pickle.dump(metadata_to_save, f)
                print(f"Metadata saved to {meta_save_path}")

                # Save training data
                if len(train_dataset) > 0:
                    np.savez_compressed(train_npz_save_path,
                                        states=train_dataset.states,
                                        actions=train_dataset.actions,
                                        rewards=train_dataset.rewards,
                                        next_states=train_dataset.next_states)
                    print(f"Training data saved to {train_npz_save_path}")
                else:
                    # Save empty structure if train_dataset is empty
                    np.savez_compressed(train_npz_save_path, states=np.array([]), actions=np.array([]), rewards=np.array([]), next_states=np.array([]))
                    print(f"Training dataset is empty. Saved empty structure to {train_npz_save_path}")


                # Save validation data
                if len(validation_dataset) > 0:
                    np.savez_compressed(val_npz_save_path,
                                        states=validation_dataset.states,
                                        actions=validation_dataset.actions,
                                        rewards=validation_dataset.rewards,
                                        next_states=validation_dataset.next_states)
                    print(f"Validation data saved to {val_npz_save_path}")
                else:
                    # Save empty structure if val_dataset is empty
                    np.savez_compressed(val_npz_save_path, states=np.array([]), actions=np.array([]), rewards=np.array([]), next_states=np.array([]))
                    print(f"Validation dataset is empty. Saved empty structure to {val_npz_save_path}")

            except Exception as e:
                print(f"Error saving dataset parts (meta: {meta_save_path}, train: {train_npz_save_path}, val: {val_npz_save_path}): {e}")
        else:
            print("No new data was collected, so dataset will not be saved.")

    return train_dataset, validation_dataset


def collect_ppo_episodes(config, max_steps_per_episode, image_size, validation_split_ratio):
    env_name = config['environment_name']
    num_episodes = config['num_episodes_data_collection']

    dataset_dir = config['dataset_dir']
    load_dataset_filename = config['load_dataset_path']
    save_dataset_filename = config['dataset_filename']

    os.makedirs(dataset_dir, exist_ok=True)

    data_loaded_successfully = False
    # Define preprocess transform here, as it's needed for both loading and new collection
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize(image_size),
        T.ToTensor()
    ])

    if load_dataset_filename:
        load_base_name = load_dataset_filename
        for ext in ['.pkl', '.npz']: # Remove known extensions
            if load_base_name.endswith(ext):
                load_base_name = load_base_name[:-len(ext)]
                break

        meta_path = os.path.join(dataset_dir, f"{load_base_name}_meta.pkl")
        train_npz_path = os.path.join(dataset_dir, f"{load_base_name}_train.npz")
        val_npz_path = os.path.join(dataset_dir, f"{load_base_name}_val.npz")

        if os.path.exists(meta_path):
            print(f"Loading PPO metadata from {meta_path}...")
            try:
                with open(meta_path, 'rb') as f:
                    metadata = pickle.load(f)

                loaded_env_name = metadata.get('environment_name')
                if loaded_env_name != env_name:
                    print(f"Error: Mismatch between loaded PPO dataset environment ('{loaded_env_name}') and config environment ('{env_name}').")
                    raise ValueError("Environment mismatch in loaded PPO dataset.")

                if metadata.get('collection_method') == 'ppo':
                    print(f"Loading PPO training data from {train_npz_path}...")
                    train_data_npz = np.load(train_npz_path)
                    train_dataset = ExperienceDataset(
                        train_data_npz['states'], train_data_npz['actions'],
                        train_data_npz['rewards'], train_data_npz['next_states'],
                        transform=preprocess)

                    validation_dataset = ExperienceDataset(np.array([]), np.array([]), np.array([]), np.array([]), transform=preprocess) # Default empty
                    if os.path.exists(val_npz_path):
                        print(f"Loading PPO validation data from {val_npz_path}...")
                        val_data_npz = np.load(val_npz_path)
                        if 'states' in val_data_npz and val_data_npz['states'].shape[0] > 0:
                            validation_dataset = ExperienceDataset(
                                val_data_npz['states'], val_data_npz['actions'],
                                val_data_npz['rewards'], val_data_npz['next_states'],
                                transform=preprocess)
                        else:
                            print("PPO validation data file found but appears empty.")
                    else:
                        print(f"PPO validation data file {val_npz_path} not found. Using empty validation dataset.")

                    print(f"Successfully loaded PPO-collected dataset for environment '{loaded_env_name}' with {metadata.get('num_episodes_collected', 'N/A')} episodes (from metadata).")
                    data_loaded_successfully = True
                    return train_dataset, validation_dataset
                else:
                    print(f"Warning: Loaded dataset from {meta_path} was not collected using PPO (method: {metadata.get('collection_method', 'unknown')}). Proceeding to collect new data with PPO.")
            except Exception as e:
                print(f"Error loading PPO dataset parts (meta: {meta_path}, train: {train_npz_path}, val: {val_npz_path}): {e}. Proceeding to PPO data collection.")
        else:
            print(f"Warning: PPO Metadata file {meta_path} not found. Proceeding to PPO data collection.")
    else:
        print("`load_dataset_path` is empty. Proceeding to PPO data collection.")

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

    # Determine observation shape (must be done after env is created, and potentially PPO agent for obs space consistency)
    _obs_sample, _ = env.reset() # Use the PPO-wrapped env for this if applicable, but here `env` is the original.
    if not (isinstance(_obs_sample, np.ndarray) and _obs_sample.dtype == np.uint8):
        if env.render_mode == 'rgb_array':
            _obs_sample = env.render()
    if not (isinstance(_obs_sample, np.ndarray) and _obs_sample.ndim >=2):
         env.close()
         raise ValueError(f"Initial PPO observation from environment {env_name} is not suitable. Shape: {_obs_sample.shape if hasattr(_obs_sample, 'shape') else 'N/A'}, dtype: {_obs_sample.dtype if hasattr(_obs_sample, 'dtype') else 'N/A'}")
    obs_shape = _obs_sample.shape

    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Discrete):
        action_shape = ()
        action_dtype = np.int_
    elif isinstance(action_space, gym.spaces.Box):
        action_shape = action_space.shape
        action_dtype = action_space.dtype # Typically float32 for PPO continuous actions
    else:
        env.close()
        raise TypeError(f"Unsupported action space type for PPO: {type(action_space)}")

    max_total_transitions = num_episodes * max_steps_per_episode

    all_states = np.zeros((max_total_transitions, *obs_shape), dtype=np.uint8)
    all_next_states = np.zeros((max_total_transitions, *obs_shape), dtype=np.uint8)
    _act_shape_for_zeros = (max_total_transitions,) + action_shape if action_shape else (max_total_transitions,)
    all_actions = np.zeros(_act_shape_for_zeros, dtype=action_dtype)
    all_rewards = np.zeros(max_total_transitions, dtype=np.float32)
    all_episode_done_flags = np.zeros(max_total_transitions, dtype=bool)

    transition_counter = 0
    actual_episodes_collected = 0
    # preprocess is already defined above

    for episode_idx in range(num_episodes):
        current_state_img, info = env.reset()
        initial_obs_is_uint8_image = isinstance(current_state_img, np.ndarray) and current_state_img.dtype == np.uint8

        if not initial_obs_is_uint8_image:
            if env.render_mode == 'rgb_array':
                current_state_img = env.render()
                if not (isinstance(current_state_img, np.ndarray) and current_state_img.dtype == np.uint8):
                    print(f"Error: env.render() did not return uint8 array for PPO ep {episode_idx+1}. Skipping.")
                    continue
            else:
                print(f"Warning: Initial PPO obs for ep {episode_idx+1} not uint8 and env not 'rgb_array'. Skipping.")
                continue

        if not (isinstance(current_state_img, np.ndarray) and current_state_img.shape == obs_shape):
            print(f"Skipping PPO episode {episode_idx+1} due to unsuitable initial state shape. Expected {obs_shape}, got {current_state_img.shape if hasattr(current_state_img, 'shape') else 'N/A'}")
            continue

        terminated = False
        truncated = False
        step_count = 0
        cumulative_reward_episode = 0.0
        episode_has_transitions = False

        while not (terminated or truncated) and step_count < max_steps_per_episode:
            if transition_counter >= max_total_transitions:
                print("Warning: Reached max_total_transitions during PPO collection. Stopping early.")
                break

            action_pred, _ = ppo_agent.predict(current_state_img, deterministic=True)
            # Ensure action is in the correct format for storage (e.g. .item() for scalar discrete)
            processed_action = action_pred
            if isinstance(action_space, gym.spaces.Discrete): # Check original space
                 processed_action = action_pred.item() # Convert numpy scalar to Python int if necessary for dtype

            next_state_img, reward, terminated, truncated, info = env.step(action_pred) # Use original prediction for env.step
            cumulative_reward_episode += reward
            next_obs_is_uint8_image = isinstance(next_state_img, np.ndarray) and next_state_img.dtype == np.uint8

            if not next_obs_is_uint8_image:
                if env.render_mode == 'rgb_array':
                    next_state_img = env.render()
                    if not (isinstance(next_state_img, np.ndarray) and next_state_img.dtype == np.uint8):
                        print(f"Error: env.render() for next_state (PPO) did not return uint8 array for ep {episode_idx+1}, step {step_count+1}. Skipping step.")
                        step_count += 1; continue

            if not (isinstance(current_state_img, np.ndarray) and current_state_img.shape == obs_shape and
                    isinstance(next_state_img, np.ndarray) and next_state_img.shape == obs_shape):
                print(f"Warning: Skipping PPO step in ep {episode_idx+1}, step {step_count+1} due to unsuitable state dims. Current: {current_state_img.shape if hasattr(current_state_img, 'shape') else 'N/A'}, Next: {next_state_img.shape if hasattr(next_state_img, 'shape') else 'N/A'}. Expected {obs_shape}")
                current_state_img = next_state_img; step_count += 1
                if not (isinstance(current_state_img, np.ndarray) and current_state_img.shape == obs_shape):
                    print("Recovery failed for PPO step. Breaking episode.")
                    break
                else: continue

            all_states[transition_counter] = current_state_img
            all_actions[transition_counter] = processed_action
            all_rewards[transition_counter] = reward
            all_next_states[transition_counter] = next_state_img
            all_episode_done_flags[transition_counter] = terminated or truncated

            current_state_img = next_state_img
            step_count += 1
            transition_counter += 1
            episode_has_transitions = True

        if episode_has_transitions:
            actual_episodes_collected +=1
        print(f"PPO Episode {episode_idx+1}/{num_episodes} finished after {step_count} steps. Cumulative Reward: {cumulative_reward_episode:.2f}. Total transitions: {transition_counter}")
        if transition_counter >= max_total_transitions: break

    env.close()

    if transition_counter == 0:
        print("No data collected with PPO. Returning empty datasets.")
        empty_states_arr = np.zeros((0, *obs_shape), dtype=np.uint8) if obs_shape else np.array([])
        _empty_act_shape = (0,) + action_shape if action_shape else (0,)
        empty_actions_arr = np.zeros(_empty_act_shape, dtype=action_dtype) if action_dtype else np.array([])
        empty_train_dataset = ExperienceDataset(empty_states_arr, empty_actions_arr, np.array([]), empty_states_arr, transform=preprocess)
        empty_val_dataset = ExperienceDataset(empty_states_arr, empty_actions_arr, np.array([]), empty_states_arr, transform=preprocess)
        return empty_train_dataset, empty_val_dataset

    final_states = all_states[:transition_counter]
    final_actions = all_actions[:transition_counter]
    final_rewards = all_rewards[:transition_counter]
    final_next_states = all_next_states[:transition_counter]
    final_done_flags = all_episode_done_flags[:transition_counter]

    episode_boundaries = np.where(final_done_flags)[0] + 1
    episode_starts = np.concatenate(([0], episode_boundaries[:-1]))
    episode_segments = [(start, end) for start, end in zip(episode_starts, episode_boundaries) if end > start]

    if not episode_segments:
        print("No valid PPO episode segments. Returning empty datasets.") # Should be caught by transition_counter check
        empty_states_arr = np.zeros((0, *obs_shape), dtype=np.uint8)
        _empty_act_shape = (0,) + action_shape if action_shape else (0,)
        empty_actions_arr = np.zeros(_empty_act_shape, dtype=action_dtype)
        empty_train_dataset = ExperienceDataset(empty_states_arr, empty_actions_arr, np.array([]), empty_states_arr, transform=preprocess)
        empty_val_dataset = ExperienceDataset(empty_states_arr, empty_actions_arr, np.array([]), empty_states_arr, transform=preprocess)
        return empty_train_dataset, empty_val_dataset

    random.shuffle(episode_segments)
    num_total_actual_episodes = len(episode_segments)
    split_idx_episodes = int((1.0 - validation_split_ratio) * num_total_actual_episodes)
    train_episode_segments = episode_segments[:split_idx_episodes]
    val_episode_segments = episode_segments[split_idx_episodes:]

    print(f"Total PPO actual episodes collected: {num_total_actual_episodes}")
    print(f"Splitting into {len(train_episode_segments)} PPO training episodes and {len(val_episode_segments)} PPO validation episodes.")

    train_indices = np.concatenate([np.arange(start, end) for start, end in train_episode_segments] if train_episode_segments else [[]]).astype(int)
    val_indices = np.concatenate([np.arange(start, end) for start, end in val_episode_segments] if val_episode_segments else [[]]).astype(int)

    train_s_np = np.zeros((0, *obs_shape), dtype=np.uint8)
    _empty_act_shape_train = (0,) + action_shape if action_shape else (0,)
    train_a_np = np.zeros(_empty_act_shape_train, dtype=action_dtype)
    train_r_np = np.array([], dtype=np.float32)
    train_ns_np = np.zeros((0, *obs_shape), dtype=np.uint8)

    if len(train_indices) > 0:
        train_s_np = final_states[train_indices]
        train_a_np = final_actions[train_indices]
        train_r_np = final_rewards[train_indices]
        train_ns_np = final_next_states[train_indices]

    val_s_np = np.zeros((0, *obs_shape), dtype=np.uint8)
    _empty_act_shape_val = (0,) + action_shape if action_shape else (0,)
    val_a_np = np.zeros(_empty_act_shape_val, dtype=action_dtype)
    val_r_np = np.array([], dtype=np.float32)
    val_ns_np = np.zeros((0, *obs_shape), dtype=np.uint8)

    if len(val_indices) > 0:
        val_s_np = final_states[val_indices]
        val_a_np = final_actions[val_indices]
        val_r_np = final_rewards[val_indices]
        val_ns_np = final_next_states[val_indices]

    train_dataset = ExperienceDataset(train_s_np, train_a_np, train_r_np, train_ns_np, transform=preprocess)
    validation_dataset = ExperienceDataset(val_s_np, val_a_np, val_r_np, val_ns_np, transform=preprocess)

    print(f"PPO Training dataset: {len(train_dataset)} transitions.")
    print(f"PPO Validation dataset: {len(validation_dataset)} transitions.")

    if not data_loaded_successfully:
        if transition_counter > 0: # Check if new data was actually collected
            num_episodes_collected_target = config['num_episodes_data_collection']
            save_base_name = save_dataset_filename
            for ext in ['.pkl', '.npz']: # Remove known extensions
                if save_base_name.endswith(ext):
                    save_base_name = save_base_name[:-len(ext)]
                    break

            meta_save_path = os.path.join(dataset_dir, f"{save_base_name}_meta.pkl")
            train_npz_save_path = os.path.join(dataset_dir, f"{save_base_name}_train.npz")
            val_npz_save_path = os.path.join(dataset_dir, f"{save_base_name}_val.npz")

            metadata_to_save = {
                'environment_name': env_name,
                'num_episodes_collected': num_episodes_collected_target, # Target num_episodes from config
                'actual_episodes_collected': actual_episodes_collected, # Actual number of episodes that yielded transitions
                'image_size': image_size,
                'max_steps_per_episode': max_steps_per_episode,
                'validation_split_ratio': validation_split_ratio,
                'num_train_transitions': len(train_dataset),
                'num_val_transitions': len(validation_dataset),
                'collection_method': 'ppo',
                'ppo_config_params': ppo_specific_config
            }

            try:
                # Save metadata
                with open(meta_save_path, 'wb') as f:
                    pickle.dump(metadata_to_save, f)
                print(f"PPO metadata saved to {meta_save_path}")

                # Save training data
                if len(train_dataset) > 0:
                    np.savez_compressed(train_npz_save_path,
                                        states=train_dataset.states,
                                        actions=train_dataset.actions,
                                        rewards=train_dataset.rewards,
                                        next_states=train_dataset.next_states)
                    print(f"PPO training data saved to {train_npz_save_path}")
                else:
                    np.savez_compressed(train_npz_save_path, states=np.array([]), actions=np.array([]), rewards=np.array([]), next_states=np.array([]))
                    print(f"PPO training dataset is empty. Saved empty structure to {train_npz_save_path}")

                # Save validation data
                if len(validation_dataset) > 0:
                    np.savez_compressed(val_npz_save_path,
                                        states=validation_dataset.states,
                                        actions=validation_dataset.actions,
                                        rewards=validation_dataset.rewards,
                                        next_states=validation_dataset.next_states)
                    print(f"PPO validation data saved to {val_npz_save_path}")
                else:
                    np.savez_compressed(val_npz_save_path, states=np.array([]), actions=np.array([]), rewards=np.array([]), next_states=np.array([]))
                    print(f"PPO validation dataset is empty. Saved empty structure to {val_npz_save_path}")

            except Exception as e:
                print(f"Error saving PPO dataset parts (meta: {meta_save_path}, train: {train_npz_save_path}, val: {val_npz_save_path}): {e}")
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
            gym.make("PongNoFrameskip-v4")
            test_env_name = "PongNoFrameskip-v4"
            print("Using PongNoFrameskip-v4 for testing data collection.")
        except gym.error.MissingEnvDependency:
            print("PongNoFrameskip-v4 not available. Skipping data_utils.py example run.")
            test_env_name = None

        if test_env_name:
            # Dummy config for testing
            dummy_config = {
                'environment_name': test_env_name,
                'num_episodes_data_collection': 5, # Small number for test
                'load_dataset': False, # Test data collection and saving
                'dataset_name': ''
            }

            # Test case 1: Collect and save
            print("\n--- Test Case 1: Collect and Save ---")
            train_d, val_d = collect_random_episodes(
                config=dummy_config,
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

            # Test case 2: Load the saved dataset
            print("\n--- Test Case 2: Load Saved Dataset ---")
            # For testing, we'll use the save_dataset_filename from the previous run.
            # This means the dummy_config for loading needs to be updated.

            # The filename used for saving in Test Case 1 will be based on the new logic if we were to run it with the modified code.
            # However, the current test code saves with f"{env_name.replace('/', '_')}_{num_episodes_collected}.pkl"
            # To make Test Case 2 work with the *current* test structure without modifying the test case logic itself too much now,
            # we'll keep the old way of determining `saved_dataset_filename` for the *test only*.
            # The actual function logic uses `config['dataset_filename']` for saving.

            # This specific part of the test may need more robust updates if the goal is to test the new load/save names directly from config.
            # For now, let's ensure the function itself is correct. The test will try to load what the *original* test code saved.

            _test_case_saved_filename = f"{test_env_name.replace('/', '_')}_{dummy_config['num_episodes_data_collection']}.pkl"

            dummy_config_load = {
                'environment_name': test_env_name,
                'num_episodes_data_collection': dummy_config['num_episodes_data_collection'],
                'dataset_dir': "datasets/", # Added to match new requirements
                'load_dataset_path': _test_case_saved_filename, # What the old test case 1 would have saved
                'dataset_filename': "test_data_save.pkl" # Name for saving if this run were to save
            }

            # Check if the file actually exists before attempting to load
            # The dataset_dir for test case 2 should also align with the new config.
            dataset_file_path = os.path.join(dummy_config_load['dataset_dir'], dummy_config_load['load_dataset_path'])
            if os.path.exists(dataset_file_path):
                train_d_loaded, val_d_loaded = collect_random_episodes(
                    config=dummy_config_load, # Pass the updated dummy_config_load
                    max_steps_per_episode=50, # These are not used when loading but function expects them
                    image_size=(64, 64),    # Same here
                    validation_split_ratio=0.4 # Same here
                )

                print(f"\n--- Loaded Training Dataset (Size: {len(train_d_loaded)}) ---")
                if len(train_d_loaded) > 0:
                    # Basic check: compare sizes with originally collected data
                    assert len(train_d_loaded) == len(train_d), "Loaded train dataset size mismatch!"
                    print("Loaded training dataset size matches original.")
                    # Deeper checks could involve comparing actual data points if necessary
                else:
                    print("Loaded training dataset is empty.")

                print(f"\n--- Loaded Validation Dataset (Size: {len(val_d_loaded)}) ---")
                if len(val_d_loaded) > 0:
                    assert len(val_d_loaded) == len(val_d), "Loaded validation dataset size mismatch!"
                    print("Loaded validation dataset size matches original.")
                else:
                    print("Loaded validation dataset is empty.")
            else:
                print(f"Dataset file {dataset_file_path} not found for Test Case 2. Skipping loading test.")

    except ImportError as e:
        print(f"Import error, likely missing a dependency for the test environment: {e}")
    except Exception as e:
        print(f"An error occurred during the example run: {e}")
        import traceback
        traceback.print_exc()
# Commenting out the old test code as per subtask, new tests are in tests/test_data_utils.py
# if __name__ == '__main__':
#     # Example usage:
#     # Ensure you have a display server if using environments like CarRacing-v2 locally without headless mode.
#     # For servers, use Xvfb: Xvfb :1 -screen 0 1024x768x24 &
#     # export DISPLAY=:1

#     print(f"Testing data collection with a sample environment...")
#     try:
#         # Attempt to use a known pixel-based environment
#         try:
#             gym.make("PongNoFrameskip-v4")
#             test_env_name = "PongNoFrameskip-v4"
#             print("Using PongNoFrameskip-v4 for testing data collection.")
#         except gym.error.MissingEnvDependency:
#             print("PongNoFrameskip-v4 not available. Skipping data_utils.py example run.")
#             test_env_name = None

#         if test_env_name:
#             # Dummy config for testing
#             dummy_config = {
#                 'environment_name': test_env_name,
#                 'num_episodes_data_collection': 5, # Small number for test
#                 'load_dataset': False, # Test data collection and saving
#                 'dataset_name': ''
#             }

#             # Test case 1: Collect and save
#             print("\n--- Test Case 1: Collect and Save ---")
#             train_d, val_d = collect_random_episodes(
#                 config=dummy_config,
#                 max_steps_per_episode=50,
#                 image_size=(64, 64),
#                 validation_split_ratio=0.4
#             )

#             print(f"\n--- Training Dataset (Size: {len(train_d)}) ---")
#             if len(train_d) > 0:
#                 train_dataloader = DataLoader(train_d, batch_size=4, shuffle=True)
#                 s_batch, a_batch, r_batch, s_next_batch = next(iter(train_dataloader))
#                 print(f"Training Sample batch shapes: States {s_batch.shape}, Actions {a_batch.shape}, Rewards {r_batch.shape}, Next States {s_next_batch.shape}")
#             else:
#                 print("Training dataset is empty.")

#             print(f"\n--- Validation Dataset (Size: {len(val_d)}) ---")
#             if len(val_d) > 0:
#                 val_dataloader = DataLoader(val_d, batch_size=4, shuffle=False)
#                 s_val_batch, a_val_batch, r_val_batch, s_next_val_batch = next(iter(val_dataloader))
#                 print(f"Validation Sample batch shapes: States {s_val_batch.shape}, Actions {a_val_batch.shape}, Rewards {r_val_batch.shape}, Next States {s_next_val_batch.shape}")
#             else:
#                 print("Validation dataset is empty.")

#             # Test case 2: Load the saved dataset
#             print("\n--- Test Case 2: Load Saved Dataset ---")
#             # For testing, we'll use the save_dataset_filename from the previous run.
#             # This means the dummy_config for loading needs to be updated.

#             # The filename used for saving in Test Case 1 will be based on the new logic if we were to run it with the modified code.
#             # However, the current test code saves with f"{env_name.replace('/', '_')}_{num_episodes_collected}.pkl"
#             # To make Test Case 2 work with the *current* test structure without modifying the test case logic itself too much now,
#             # we'll keep the old way of determining `saved_dataset_filename` for the *test only*.
#             # The actual function logic uses `config['dataset_filename']` for saving.

#             # This specific part of the test may need more robust updates if the goal is to test the new load/save names directly from config.
#             # For now, let's ensure the function itself is correct. The test will try to load what the *original* test code saved.

#             _test_case_saved_filename = f"{test_env_name.replace('/', '_')}_{dummy_config['num_episodes_data_collection']}.pkl"

#             dummy_config_load = {
#                 'environment_name': test_env_name,
#                 'num_episodes_data_collection': dummy_config['num_episodes_data_collection'],
#                 'dataset_dir': "datasets/", # Added to match new requirements
#                 'load_dataset_path': _test_case_saved_filename, # What the old test case 1 would have saved
#                 'dataset_filename': "test_data_save.pkl" # Name for saving if this run were to save
#             }

#             # Check if the file actually exists before attempting to load
#             # The dataset_dir for test case 2 should also align with the new config.
#             dataset_file_path = os.path.join(dummy_config_load['dataset_dir'], dummy_config_load['load_dataset_path'])
#             if os.path.exists(dataset_file_path):
#                 train_d_loaded, val_d_loaded = collect_random_episodes(
#                     config=dummy_config_load, # Pass the updated dummy_config_load
#                     max_steps_per_episode=50, # These are not used when loading but function expects them
#                     image_size=(64, 64),    # Same here
#                     validation_split_ratio=0.4 # Same here
#                 )

#                 print(f"\n--- Loaded Training Dataset (Size: {len(train_d_loaded)}) ---")
#                 if len(train_d_loaded) > 0:
#                     # Basic check: compare sizes with originally collected data
#                     assert len(train_d_loaded) == len(train_d), "Loaded train dataset size mismatch!"
#                     print("Loaded training dataset size matches original.")
#                     # Deeper checks could involve comparing actual data points if necessary
#                 else:
#                     print("Loaded training dataset is empty.")

#                 print(f"\n--- Loaded Validation Dataset (Size: {len(val_d_loaded)}) ---")
#                 if len(val_d_loaded) > 0:
#                     assert len(val_d_loaded) == len(val_d), "Loaded validation dataset size mismatch!"
#                     print("Loaded validation dataset size matches original.")
#                 else:
#                     print("Loaded validation dataset is empty.")
#             else:
#                 print(f"Dataset file {dataset_file_path} not found for Test Case 2. Skipping loading test.")

#     except ImportError as e:
#         print(f"Import error, likely missing a dependency for the test environment: {e}")
#     except Exception as e:
#         print(f"An error occurred during the example run: {e}")
#         import traceback
#         traceback.print_exc()
