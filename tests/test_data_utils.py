import unittest
import os
import sys
import gymnasium as gym
import math
import pickle
import shutil
import torch # For creating dummy ExperienceDataset if needed for dummy file

# Add project root to sys.path to allow importing project modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_utils import collect_random_episodes, ExperienceDataset

# Try Pong first, fallback to CartPole if Atari ROMs are an issue for testing CI
try:
    gym.make("PongNoFrameskip-v4").close()
    ENV_NAME = "PongNoFrameskip-v4"
    print(f"Using {ENV_NAME} for data_utils tests.")
    ENV_AVAILABLE = True
except Exception:
    print(f"PongNoFrameskip-v4 not available. Falling back to CartPole-v1 for data_utils tests.")
    try:
        gym.make("CartPole-v1").close()
        ENV_NAME = "CartPole-v1"
        ENV_AVAILABLE = True
    except Exception:
        ENV_NAME = "CartPole-v1" # Default for skip message
        ENV_AVAILABLE = False

TEST_DATASET_DIR = "test_datasets_temp"


@unittest.skipIf(not ENV_AVAILABLE, f"Base environment {ENV_NAME} (or fallback) not available. Skipping data_utils tests.")
class TestDataUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs(TEST_DATASET_DIR, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TEST_DATASET_DIR, ignore_errors=True)

    def _get_base_config(self, num_episodes, load_dataset=False, dataset_name=""):
        # In config.yaml, image_size is an int. data_handling.py converts it to a tuple.
        # collect_random_episodes expects a tuple for image_size argument, but gets config dict.
        # The test needs to align with what collect_random_episodes expects from config for its own internal processing
        # versus what it expects as a direct argument.
        # For the 'image_size' key *in the config object*, it's typically an int.
        # The image_size *parameter* to collect_random_episodes is a tuple.
        return {
            'environment_name': ENV_NAME,
            'num_episodes_data_collection': num_episodes,
            'load_dataset': load_dataset,
            'dataset_name': dataset_name,
            'image_size': 64, # As it would be in config.yaml
            'dataset_output_dir_override': TEST_DATASET_DIR # For tests
        }

    def test_collect_random_episodes_split(self):
        num_episodes_total = 10
        if ENV_NAME == "CartPole-v1": # CartPole is very fast, can do more
             num_episodes_total = 20
        elif "Pong" in ENV_NAME: # Pong is slower
            num_episodes_total = 4 # Reduced for Pong to speed up test

        max_steps_per_episode = 5
        image_size_tuple = (64, 64) # This is passed as direct arg, not from config

        print(f"\nStarting test_collect_random_episodes_split with {ENV_NAME}...")

        base_config = self._get_base_config(num_episodes_total)

        # Test Case 1: 20% validation split
        val_split_ratio_1 = 0.2
        expected_val_episodes_1 = int(num_episodes_total * val_split_ratio_1)
        expected_train_episodes_1 = num_episodes_total - expected_val_episodes_1

        print(f"\nTest Case 1: val_split_ratio={val_split_ratio_1}")
        train_dataset_1, val_dataset_1 = collect_random_episodes(
            base_config, max_steps_per_episode, image_size_tuple, val_split_ratio_1
        )

        self.assertIsInstance(train_dataset_1, ExperienceDataset)
        self.assertIsInstance(val_dataset_1, ExperienceDataset)

        if expected_train_episodes_1 > 0:
            self.assertTrue(len(train_dataset_1) >= expected_train_episodes_1 * 1)
            self.assertTrue(len(train_dataset_1) <= expected_train_episodes_1 * max_steps_per_episode)
        else:
            self.assertEqual(len(train_dataset_1), 0)

        if expected_val_episodes_1 > 0:
            self.assertTrue(len(val_dataset_1) >= expected_val_episodes_1 * 1)
            self.assertTrue(len(val_dataset_1) <= expected_val_episodes_1 * max_steps_per_episode)
        else:
            self.assertEqual(len(val_dataset_1), 0)

        # Test Case 2: 0% validation split
        val_split_ratio_2 = 0.0
        expected_train_episodes_2 = num_episodes_total

        print(f"\nTest Case 2: val_split_ratio={val_split_ratio_2}")
        train_dataset_2, val_dataset_2 = collect_random_episodes(
            base_config, max_steps_per_episode, image_size_tuple, val_split_ratio_2
        )
        self.assertTrue(len(train_dataset_2) >= expected_train_episodes_2 * 1)
        self.assertTrue(len(train_dataset_2) <= expected_train_episodes_2 * max_steps_per_episode)
        self.assertEqual(len(val_dataset_2), 0)

        # Test Case 3: 100% validation split
        val_split_ratio_3 = 1.0
        expected_val_episodes_3 = num_episodes_total

        print(f"\nTest Case 3: val_split_ratio={val_split_ratio_3}")
        train_dataset_3, val_dataset_3 = collect_random_episodes(
            base_config, max_steps_per_episode, image_size_tuple, val_split_ratio_3
        )
        self.assertEqual(len(train_dataset_3), 0)
        self.assertTrue(len(val_dataset_3) >= expected_val_episodes_3 * 1)
        self.assertTrue(len(val_dataset_3) <= expected_val_episodes_3 * max_steps_per_episode)

        print("\nFinished test_collect_random_episodes_split.")

    def test_save_and_load_dataset(self):
        num_episodes = 2
        if "Pong" in ENV_NAME:
            num_episodes = 2 # Keep it small for Pong
        max_steps = 5
        img_size_tuple = (64, 64)
        val_split = 0.5 # Ensure both datasets get some data

        collect_config = self._get_base_config(num_episodes, load_dataset=False)

        print(f"\nStarting test_save_and_load_dataset with {ENV_NAME}...")
        print("Collecting and saving dataset first...")
        train_ds_orig, val_ds_orig = collect_random_episodes(
            collect_config, max_steps, img_size_tuple, val_split
        )

        self.assertTrue(len(train_ds_orig) > 0, "Original training dataset is empty after collection.")
        self.assertTrue(len(val_ds_orig) > 0, "Original validation dataset is empty after collection.")

        sanitized_env_name = ENV_NAME.replace('/', '_')
        expected_filename = f"{sanitized_env_name}_{num_episodes}.pkl"
        expected_filepath = os.path.join(TEST_DATASET_DIR, expected_filename)
        self.assertTrue(os.path.exists(expected_filepath), f"Dataset file {expected_filepath} was not created.")

        print(f"Dataset saved. Now attempting to load {expected_filename}...")

        load_config = self._get_base_config(num_episodes, load_dataset=True, dataset_name=expected_filename)

        train_ds_loaded, val_ds_loaded = collect_random_episodes(
            load_config, max_steps, img_size_tuple, val_split # other args shouldn't matter for loading
        )

        self.assertEqual(len(train_ds_loaded), len(train_ds_orig), "Loaded train dataset length mismatch.")
        self.assertEqual(len(val_ds_loaded), len(val_ds_orig), "Loaded validation dataset length mismatch.")

        # Optionally, compare actual data if ExperienceDataset supports equality or item access
        if len(train_ds_orig) > 0 and len(train_ds_loaded) > 0:
             s1o, a1o, r1o, ns1o = train_ds_orig[0]
             s1l, a1l, r1l, ns1l = train_ds_loaded[0]
             self.assertTrue(torch.equal(s1o,s1l), "First state tensor in loaded train dataset does not match original.")
             # Action comparison might need tolerance if float
             self.assertTrue(torch.allclose(a1o, a1l), "First action tensor in loaded train dataset does not match original.")


        print("Finished test_save_and_load_dataset.")

    def test_load_dataset_env_mismatch(self):
        num_episodes = 1
        max_steps = 2
        img_size_tuple = (64,64)
        val_split = 0.0 # Not critical here

        dummy_env_name_orig = "FakeEnv-TotallyDifferent-v0"
        sanitized_dummy_env_name = dummy_env_name_orig.replace('/', '_')
        dummy_file_name = f"{sanitized_dummy_env_name}_{num_episodes}.pkl"
        dummy_dataset_path = os.path.join(TEST_DATASET_DIR, dummy_file_name)

        # Create a dummy dataset file with different environment metadata
        dummy_metadata = {'environment_name': dummy_env_name_orig, 'num_episodes_collected': num_episodes}
        # Create minimal ExperienceDataset instances
        dummy_train_ds = ExperienceDataset([], [], [], [], transform=None)
        dummy_val_ds = ExperienceDataset([], [], [], [], transform=None)
        data_to_save = {'train_dataset': dummy_train_ds, 'val_dataset': dummy_val_ds, 'metadata': dummy_metadata}

        with open(dummy_dataset_path, 'wb') as f:
            pickle.dump(data_to_save, f)

        self.assertTrue(os.path.exists(dummy_dataset_path))

        print(f"\nStarting test_load_dataset_env_mismatch. Actual ENV_NAME: {ENV_NAME}, Dummy file's env: {dummy_env_name_orig}")

        # Prepare config for loading, but with ENV_NAME (which is different from dummy_env_name_orig)
        mismatch_config = self._get_base_config(num_episodes, load_dataset=True, dataset_name=dummy_file_name)

        with self.assertRaisesRegex(ValueError, "Environment mismatch in loaded dataset."):
            collect_random_episodes(mismatch_config, max_steps, img_size_tuple, val_split)

        print("Finished test_load_dataset_env_mismatch.")

    def test_load_dataset_file_not_found_collects_new(self):
        num_episodes_to_collect = 2 # Should collect these if file not found
        if "Pong" in ENV_NAME: num_episodes_to_collect = 1

        max_steps = 3
        img_size_tuple = (64,64)
        val_split = 0.5

        non_existent_filename = "this_dataset_does_not_exist_ever.pkl"

        not_found_config = self._get_base_config(
            num_episodes_to_collect,
            load_dataset=True,
            dataset_name=non_existent_filename
        )

        print(f"\nStarting test_load_dataset_file_not_found_collects_new with {ENV_NAME}...")
        # Expect a warning to be printed by the function, then new data collection.
        train_ds, val_ds = collect_random_episodes(
            not_found_config, max_steps, img_size_tuple, val_split
        )

        # Check that new data was collected
        self.assertTrue(len(train_ds) > 0 or len(val_ds) > 0, "No data collected after non-existent dataset load attempt.")

        # Verify that the newly collected data corresponds to num_episodes_to_collect
        # num_train_episodes + num_val_episodes should be num_episodes_to_collect
        # (This logic is internal to collect_random_episodes, so we check combined length)
        # Each episode gives at least 1 transition.
        total_transitions = len(train_ds) + len(val_ds)
        self.assertTrue(total_transitions >= num_episodes_to_collect * 1)
        self.assertTrue(total_transitions <= num_episodes_to_collect * max_steps)

        # And a new file should have been saved with the *correct* environment name
        sanitized_env_name = ENV_NAME.replace('/', '_')
        newly_saved_filename = f"{sanitized_env_name}_{num_episodes_to_collect}.pkl"
        newly_saved_filepath = os.path.join(TEST_DATASET_DIR, newly_saved_filename)
        self.assertTrue(os.path.exists(newly_saved_filepath), f"New dataset file {newly_saved_filepath} was not created after failed load.")
        print("Finished test_load_dataset_file_not_found_collects_new.")


if __name__ == '__main__':
    unittest.main()
