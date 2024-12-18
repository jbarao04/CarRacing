import os
import gymnasium as gym
from stable_baselines3 import DDPG, PPO, DQN, A2C
from car_racing_modified import CarRacing
#from car_racing_original import CarRacing
import sb3_contrib
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation

models_dir = "models/DDPG"
logdir="logs"

# Create the environment
env = CarRacing()
env = ResizeObservation(env, shape=(64,64))
env= GrayscaleObservation(env,keep_dim=False) 
env = FrameStackObservation(env, stack_size=2)
env.reset()

# Load the latest model if it exists, or start a new one
model_path = f"{models_dir}/40000000.zip"  # Update this with the path to your last saved model
if os.path.exists(model_path):
    print("Loading the saved model...")
    model = DDPG.load(model_path, env=env)  # Make sure to pass the environment when loading
else:
    print("No saved model found, starting a new model...")
    model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
TIMESTEPS = 100000
for i in range (1,101):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="logs_DDPG")  
    model.save(f"{models_dir}/{0 + i * TIMESTEPS}")