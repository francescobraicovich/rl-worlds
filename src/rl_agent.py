# src/rl_agent.py
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def create_ppo_agent(env, ppo_config: dict, device: str = 'mps'):
    """
    Creates a PPO agent.

    Args:
        env: The Gym environment (non-vectorized).
        ppo_config: Dictionary containing PPO parameters like
                    learning_rate, n_steps, batch_size, n_epochs,
                    gamma, gae_lambda, clip_range, policy_type.
        device: The device to train on ('auto', 'cpu', 'cuda').

    Returns:
        A PPO agent.
    """

    # Wrap the environment in a DummyVecEnv for SB3
    # SB3 typically expects a vectorized environment
    vec_env = DummyVecEnv([lambda: env])

    agent = PPO(
        ppo_config.get('policy_type', 'MlpPolicy'), # Default to MlpPolicy if not specified
        vec_env,
        learning_rate=ppo_config.get('learning_rate', 0.0003),
        n_steps=ppo_config.get('n_steps', 2048),
        batch_size=ppo_config.get('batch_size', 64),
        n_epochs=ppo_config.get('n_epochs', 10),
        gamma=ppo_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_range=ppo_config.get('clip_range', 0.2),
        tensorboard_log=None, # Can be set to a path to log training
        verbose=1, # Set to 0 for less output, 1 for info
        device=device,
        # policy_kwargs will be passed to the policy network constructor
        # e.g., policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]) for MLP
        # For CnnPolicy, features_extractor_kwargs can be used if needed.
    )
    return agent

def train_ppo_agent(agent: PPO, ppo_config: dict, task_name: str = "PPO Training"):
    """
    Trains the PPO agent.

    Args:
        agent: The PPO agent to train.
        ppo_config: Dictionary containing PPO parameters, primarily 'total_train_timesteps'.
        task_name: A descriptive name for the training task for print statements.
    """
    total_timesteps = ppo_config.get('total_train_timesteps', 100000)
    print(f"Starting {task_name} for {total_timesteps} timesteps...")
    agent.learn(total_timesteps=total_timesteps, progress_bar=True)
    print(f"{task_name} complete.")

if __name__ == '__main__':
    # This is a placeholder for testing the rl_agent.py script independently
    # You would need a Gym environment and a sample config to run this.
    print("Testing rl_agent.py script...")

    # Example of how you might test this (requires a Gym environment and config)
    import gymnasium as gym
    from src.utils.config_utils import load_config # Assuming this utility exists

    try:
        # Load a sample configuration (replace with your actual config path)
        # This assumes config.yaml is in the parent directory and has a ppo_agent section
        # Adjust the path as necessary depending on where you run this test from.
        config_path = '../config.yaml' # Path relative to src/ if running from src/
        if not torch.cuda.is_available(): # Simple check, adjust path if needed
            # If running from root, path would be 'config.yaml'
            # This is a heuristic, better to use absolute paths or more robust relative path logic
             config_path = 'config.yaml'


        full_config = load_config(config_path)
        ppo_specific_config = full_config.get('ppo_agent', {})

        if not ppo_specific_config:
            print("PPO configuration not found in config.yaml. Skipping test.")
        else:
            print(f"PPO Config found: {ppo_specific_config}")

            # Create a dummy environment for testing
            # Replace 'CartPole-v1' with an environment that matches your PPO policy_type
            # e.g., if policy_type is 'CnnPolicy', use an image-based environment
            env_name_test = full_config.get('environment_name', 'CartPole-v1')
            print(f"Creating test environment: {env_name_test}")

            # Determine if the environment is image-based to select appropriate default policy
            is_image_env = "CarRacing" in env_name_test or "Pong" in env_name_test or "ALE/" in env_name_test # Add other image envs
            default_policy = "CnnPolicy" if is_image_env else "MlpPolicy"
            if 'policy_type' not in ppo_specific_config:
                 ppo_specific_config['policy_type'] = default_policy
                 print(f"Policy type not in PPO config, defaulting to {default_policy} based on env name.")

            try:
                # For image-based environments, render_mode='rgb_array' is often needed
                # For non-image based, it might not be supported or necessary.
                if is_image_env:
                    test_env = gym.make(env_name_test, render_mode='rgb_array')
                else:
                    test_env = gym.make(env_name_test)
                test_env = Monitor(test_env) # Wrap with Monitor
                print(f"Test environment '{env_name_test}' created successfully and wrapped with Monitor.")
            except Exception as e:
                print(f"Could not create test environment '{env_name_test}'. Error: {e}")
                print("Skipping rl_agent.py test.")
                test_env = None


            if test_env:
                # Determine device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {device}")

                # Create agent
                print("Creating PPO agent...")
                agent = create_ppo_agent(test_env, ppo_specific_config, device=device)
                print("PPO agent created.")

                # Train agent (using a very small number of timesteps for a quick test)
                test_train_config = ppo_specific_config.copy()
                test_train_config['total_train_timesteps'] = 128 # Minimal timesteps for testing
                test_train_config['n_steps'] = 32 # Ensure n_steps is small for quick test

                print("Training PPO agent (test)...")
                train_ppo_agent(agent, test_train_config, task_name="PPO Test Training")
                print("PPO agent training (test) complete.")

                # Example of getting an action (requires an observation)
                obs, _ = test_env.reset()
                action, _ = agent.predict(obs, deterministic=True)
                print(f"Agent predicted action: {action}")

                test_env.close()
                print("Test environment closed.")
                print("rl_agent.py test completed.")

    except FileNotFoundError:
        print(f"Config file for testing not found at guessed path. Please ensure '{config_path}' exists or adjust the path.")
    except ImportError as e:
        print(f"Import error during testing: {e}. Make sure all dependencies are installed.")
    except Exception as e:
        print(f"An error occurred during rl_agent.py self-test: {e}")
        import traceback
        traceback.print_exc()
