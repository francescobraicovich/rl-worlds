import unittest
import os
import tempfile
import shutil
import pickle
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from src.utils.data_utils import collect_random_episodes, collect_ppo_episodes, ExperienceDataset
from torchvision import transforms as T

# Mock Gym Environment for testing
class MockImageEnv(gym.Env):
    def __init__(self, obs_height=64, obs_width=64, n_channels=3, n_actions=2):
        super(MockImageEnv, self).__init__()
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.n_channels = n_channels
        self.n_actions = n_actions

        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(obs_height, obs_width, n_channels),
            dtype=np.uint8
        )
        self.current_step = 0
        # Make max_steps_per_episode configurable for different test needs
        self.max_episode_steps_config = 10


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # If options specify max_steps, use it for this episode
        if options and 'max_episode_steps' in options:
            self.current_max_steps = options['max_episode_steps']
        else:
            self.current_max_steps = self.max_episode_steps_config

        obs = self.observation_space.sample()
        info = {}
        return obs, info

    def step(self, action):
        self.current_step += 1
        obs = self.observation_space.sample()
        reward = float(self.np_random.random())
        terminated = self.current_step >= self.current_max_steps
        truncated = False # Not typically used in simple envs unless time limit is external
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self): # In 'rgb_array' mode, render() is the source of truth for images
        return self.observation_space.sample()

    def close(self):
        pass

# Try to register only once
try:
    gym.make('MockImageEnv-v0')
except gym.error.NameNotFound:
    gym.register(
        id='MockImageEnv-v0',
        entry_point='tests.test_data_utils:MockImageEnv', # Path for test runner
        kwargs={'obs_height':64, 'obs_width':64, 'n_channels':3, 'n_actions':2}
    )


class TestDataUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure the environment is registered before any tests run
        # This is important if tests are run in parallel or out of order
        try:
            gym.make('MockImageEnv-v0')
        except gym.error.NameNotFound:
            gym.register(
                id='MockImageEnv-v0',
                entry_point='tests.test_data_utils:MockImageEnv',
                 kwargs={'obs_height':64, 'obs_width':64, 'n_channels':3, 'n_actions':2}
            )


    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.temp_dir, "datasets")
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.mock_env_name = 'MockImageEnv-v0'
        self.mock_env_instance = gym.make(self.mock_env_name) # To get default shapes for comparison

        self.image_size = (32, 32) # Smaller than mock env's native for resize test
        # Standard PyTorch transform for images (C, H, W)
        self.pytorch_transform = T.Compose([T.ToPILImage(), T.Resize(self.image_size), T.ToTensor()])

        # Expected shape after PyTorch transform (C, H, W)
        # Mock env has 3 channels by default.
        self.expected_processed_state_shape = (self.mock_env_instance.observation_space.shape[2], *self.image_size)


        self.base_config = {
            'environment_name': self.mock_env_name,
            'dataset_dir': self.dataset_dir,
            'dataset_filename': "test_data.pkl",
            'load_dataset_path': "",
            'num_episodes_data_collection': 2,
            'ppo_agent': {
                'enabled': True,
                'total_train_timesteps': 10,
                'n_steps': 5, # Must be <= total_train_timesteps
                'batch_size': 5,
                'n_epochs': 1,
                'policy_kwargs': {
                    'features_extractor_kwargs': {'features_dim': 32},
                    'net_arch': {'pi': [32], 'vf': [32]}
                }
            }
        }


    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        self.mock_env_instance.close()

    def _compare_datasets(self, ds1, ds2, test_getitem=True, num_items_to_check=3, expected_item_state_shape=None):
        self.assertEqual(len(ds1), len(ds2), "Dataset lengths differ.")
        if len(ds1) == 0:
            self.assertEqual(ds1.states.ndim, ds2.states.ndim, "Empty states ndim differ") # Should be at least 1 for shape[0]
            # For empty states, expect shape like (0, H, W, C) or (0,) for actions
            self.assertTrue(ds1.states.shape[0] == 0 and ds2.states.shape[0] == 0)
            self.assertTrue(ds1.actions.shape[0] == 0 and ds2.actions.shape[0] == 0)
            return

        self.assertTrue(np.array_equal(ds1.states, ds2.states), "Raw states differ.")
        self.assertTrue(np.array_equal(ds1.actions, ds2.actions), "Raw actions differ.")
        self.assertTrue(np.allclose(ds1.rewards, ds2.rewards), "Raw rewards differ.")
        self.assertTrue(np.array_equal(ds1.next_states, ds2.next_states), "Raw next_states differ.")

        if test_getitem:
            indices_to_check = np.random.choice(len(ds1), min(num_items_to_check, len(ds1)), replace=False)
            for idx in indices_to_check:
                s1, a1, r1, ns1 = ds1[idx]
                s2, a2, r2, ns2 = ds2[idx]

                self.assertTrue(torch.equal(s1, s2), f"State tensors differ at index {idx}")
                self.assertTrue(torch.equal(a1, a2), f"Action tensors differ at index {idx}") # Actions are float32 in dataset
                self.assertTrue(torch.allclose(r1, r2), f"Reward tensors differ at index {idx}")
                self.assertTrue(torch.equal(ns1, ns2), f"Next state tensors differ at index {idx}")

                if expected_item_state_shape:
                    self.assertEqual(s1.shape, expected_item_state_shape, f"State tensor shape incorrect at index {idx}")
                    self.assertEqual(ns1.shape, expected_item_state_shape, f"Next state tensor shape incorrect at index {idx}")


    def test_collect_random_save_load(self):
        config = self.base_config.copy()
        max_steps_per_ep = 7 # Use a prime number to better test splits
        config['num_episodes_data_collection'] = 3

        orig_train_ds, orig_val_ds = collect_random_episodes(
            config, max_steps_per_episode=max_steps_per_ep,
            image_size=self.image_size, validation_split_ratio=0.4 # Approx 60/40 split
        )

        base_name = config['dataset_filename'].replace(".pkl","").replace(".npz","")
        meta_path = os.path.join(self.dataset_dir, f"{base_name}_meta.pkl")
        train_npz_path = os.path.join(self.dataset_dir, f"{base_name}_train.npz")
        val_npz_path = os.path.join(self.dataset_dir, f"{base_name}_val.npz")

        self.assertTrue(os.path.exists(meta_path))
        self.assertTrue(os.path.exists(train_npz_path))
        self.assertTrue(os.path.exists(val_npz_path))

        with open(meta_path, 'rb') as f:
            saved_metadata = pickle.load(f)

        self.assertEqual(saved_metadata['environment_name'], config['environment_name'])
        self.assertEqual(saved_metadata['collection_method'], 'random')
        self.assertEqual(saved_metadata['actual_episodes_collected'], config['num_episodes_data_collection'])


        config['load_dataset_path'] = config['dataset_filename']
        loaded_train_ds, loaded_val_ds = collect_random_episodes(
            config, max_steps_per_episode=max_steps_per_ep,
            image_size=self.image_size, validation_split_ratio=0.4
        )

        self.assertIsNotNone(loaded_train_ds)
        self.assertIsNotNone(loaded_val_ds)
        self._compare_datasets(orig_train_ds, loaded_train_ds, expected_item_state_shape=self.expected_processed_state_shape)
        self._compare_datasets(orig_val_ds, loaded_val_ds, expected_item_state_shape=self.expected_processed_state_shape)

        with open(meta_path, 'rb') as f:
            loaded_metadata_check = pickle.load(f)
        self.assertEqual(loaded_metadata_check['num_train_transitions'], len(loaded_train_ds))
        self.assertEqual(loaded_metadata_check['num_val_transitions'], len(loaded_val_ds))
        self.assertEqual(len(orig_train_ds) + len(orig_val_ds), config['num_episodes_data_collection'] * max_steps_per_ep)


    def test_collect_ppo_save_load(self):
        config = self.base_config.copy()
        config['ppo_agent']['enabled'] = True
        config['ppo_agent']['total_train_timesteps'] = 10
        config['ppo_agent']['n_steps'] = 5
        config['ppo_agent']['batch_size'] = 5
        max_steps_per_ep = 6
        config['num_episodes_data_collection'] = 2

        orig_train_ds, orig_val_ds = collect_ppo_episodes(
            config, max_steps_per_episode=max_steps_per_ep,
            image_size=self.image_size, validation_split_ratio=0.5
        )

        base_name = config['dataset_filename'].replace(".pkl","").replace(".npz","")
        meta_path = os.path.join(self.dataset_dir, f"{base_name}_meta.pkl")
        self.assertTrue(os.path.exists(meta_path))

        with open(meta_path, 'rb') as f:
            saved_metadata = pickle.load(f)
        self.assertEqual(saved_metadata['collection_method'], 'ppo')
        self.assertIn('ppo_config_params', saved_metadata)
        self.assertEqual(saved_metadata['actual_episodes_collected'], config['num_episodes_data_collection'])


        config['load_dataset_path'] = config['dataset_filename']
        loaded_train_ds, loaded_val_ds = collect_ppo_episodes(
            config, max_steps_per_episode=max_steps_per_ep,
            image_size=self.image_size, validation_split_ratio=0.5
        )

        self.assertIsNotNone(loaded_train_ds)
        self.assertIsNotNone(loaded_val_ds)
        self._compare_datasets(orig_train_ds, loaded_train_ds, expected_item_state_shape=self.expected_processed_state_shape)
        self._compare_datasets(orig_val_ds, loaded_val_ds, expected_item_state_shape=self.expected_processed_state_shape)
        self.assertEqual(len(orig_train_ds) + len(orig_val_ds), config['num_episodes_data_collection'] * max_steps_per_ep)


    def test_empty_dataset_random(self):
        config = self.base_config.copy()
        config['num_episodes_data_collection'] = 0
        max_steps_per_ep = 5

        orig_train_ds, orig_val_ds = collect_random_episodes(
            config, max_steps_per_episode=max_steps_per_ep,
            image_size=self.image_size, validation_split_ratio=0.5
        )
        self.assertEqual(len(orig_train_ds), 0)
        self.assertEqual(len(orig_val_ds), 0)
        self.assertEqual(orig_train_ds.states.shape, (0, *self.mock_env_instance.observation_space.shape))
        self.assertEqual(orig_train_ds.actions.shape, (0,) if not self.mock_env_instance.action_space.shape else (0, *self.mock_env_instance.action_space.shape))


        base_name = config['dataset_filename'].replace(".pkl","").replace(".npz","")
        meta_path = os.path.join(self.dataset_dir, f"{base_name}_meta.pkl")
        train_npz_path = os.path.join(self.dataset_dir, f"{base_name}_train.npz")
        val_npz_path = os.path.join(self.dataset_dir, f"{base_name}_val.npz")

        self.assertTrue(os.path.exists(meta_path))
        self.assertTrue(os.path.exists(train_npz_path))
        self.assertTrue(os.path.exists(val_npz_path))

        with open(meta_path, 'rb') as f:
            saved_metadata = pickle.load(f)
        self.assertEqual(saved_metadata['num_train_transitions'], 0)
        self.assertEqual(saved_metadata['num_val_transitions'], 0)
        self.assertEqual(saved_metadata['actual_episodes_collected'], 0)


        config['load_dataset_path'] = config['dataset_filename']
        loaded_train_ds, loaded_val_ds = collect_random_episodes(
            config, max_steps_per_episode=max_steps_per_ep,
            image_size=self.image_size, validation_split_ratio=0.5
        )
        self.assertEqual(len(loaded_train_ds), 0)
        self.assertEqual(len(loaded_val_ds), 0)
        self.assertEqual(loaded_train_ds.states.shape, (0, *self.mock_env_instance.observation_space.shape))
        self.assertEqual(loaded_train_ds.actions.shape, (0,) if not self.mock_env_instance.action_space.shape else (0, *self.mock_env_instance.action_space.shape))


    def test_validation_split_random(self):
        config = self.base_config.copy()
        num_eps_collect = 4
        max_steps_per_ep = self.mock_env_instance.max_episode_steps_config # Use mock env's default
        config['num_episodes_data_collection'] = num_eps_collect
        total_expected_transitions = num_eps_collect * max_steps_per_ep

        # Case 1: validation_split_ratio = 0.0
        train_ds, val_ds = collect_random_episodes(
            config, max_steps_per_episode=max_steps_per_ep,
            image_size=self.image_size, validation_split_ratio=0.0
        )
        self.assertEqual(len(val_ds), 0)
        self.assertEqual(len(train_ds), total_expected_transitions)

        # Case 2: validation_split_ratio = 1.0
        train_ds, val_ds = collect_random_episodes(
            config, max_steps_per_episode=max_steps_per_ep,
            image_size=self.image_size, validation_split_ratio=1.0
        )
        self.assertEqual(len(train_ds), 0)
        self.assertEqual(len(val_ds), total_expected_transitions)

        # Case 3: validation_split_ratio = 0.5
        train_ds, val_ds = collect_random_episodes(
            config, max_steps_per_episode=max_steps_per_ep,
            image_size=self.image_size, validation_split_ratio=0.5
        )
        self.assertEqual(len(train_ds) + len(val_ds), total_expected_transitions)
        # With episode shuffling, it should be an exact split of episodes.
        # Since mock env has fixed episode length, transitions should also split exactly.
        self.assertEqual(len(train_ds), (num_eps_collect // 2) * max_steps_per_ep)
        self.assertEqual(len(val_ds), (num_eps_collect - (num_eps_collect // 2)) * max_steps_per_ep)

if __name__ == '__main__':
    unittest.main()
