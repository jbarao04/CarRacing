import os
import gymnasium as gym  # Only import gymnasium
from stable_baselines3 import PPO, DDPG, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback
from car_racing_original import CarRacing  # Ensure your custom environment is imported correctly
from gymnasium.wrappers import Monitor

models_dir = "models/MLP_Models"
logdir = "logs_MLP_Models"

# Create the directories if they don't exist
if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Create the environment and wrap it with the Monitor
monitor_dir = "monitor_logs"
os.makedirs(monitor_dir, exist_ok=True)

# Create and wrap the training environment
env = CarRacing(continuous=False)  # Replace with your custom environment
env = Monitor(env, directory=monitor_dir, force=True)  # Wrap training environment with Monitor

# Create and wrap the evaluation environment (it's recommended to have a separate evaluation environment)
eval_env = CarRacing(continuous=False)  # Replace with your custom environment
eval_env = Monitor(eval_env, directory=monitor_dir, force=True)  # Wrap evaluation environment with Monitor

# Reset the environment
env.reset()

# Setup the evaluation callback with the eval_env wrapped in Monitor
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{models_dir}/best_model",
    log_path=logdir,
    eval_freq=50000,  # Evaluation frequency
    deterministic=True,
    render=False,
)

# Define the models to be used
models = {
    "PPO": PPO,
    "DDPG": DDPG,
    "DQN": DQN,
    "A2C": A2C,
}

# Function to train each model
def train_model(model_name, model_class):
    print(f"Training {model_name}...")

    # Load the latest model if it exists, or start a new one
    model_path = f"{models_dir}/{model_name}/100000.zip"  # Update with correct path for last saved model
    if os.path.exists(model_path):
        print(f"Loading the saved {model_name} model...")
        model = model_class.load(model_path, env=env)
    else:
        print(f"No saved {model_name} model found, starting a new model...")
        model = model_class(
            "MlpPolicy",  # Use MLP policy for all models
            env,
            verbose=1,
            tensorboard_log=logdir,
            ent_coef=0.02,  # Exploration factor
            gamma=0.995,  # Focus more on future rewards
            learning_rate=1e-4,  # Stable learning
        )

    # Training loop for each model
    TIMESTEPS = 100000
    for i in range(1, 11):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=model_name, callback=eval_callback)
        model.save(f"{models_dir}/{model_name}/{i * TIMESTEPS}")

# Train each model
for model_name, model_class in models.items():
    train_model(model_name, model_class)

# Example of testing after training
'''env = CarRacing(render_mode="human")
env.reset()
model_path = f"{models_dir}/PPO/500000.zip"  # Update path to any model you want to test
model = PPO.load(model_path, env=env)

episodes = 10
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)

env.close()'''
