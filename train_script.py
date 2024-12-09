import os
import gymnasium as gym
from stable_baselines3 import PPO, DDPG, DQN, A2C
#from car_racing_modified import CarRacing
from car_racing_original import CarRacing

#models_dir = "models/PPO"
models_dir = "models/PPO_Original"
logdir = "logs_original"
#logdir = "logs"


if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# Create the environment
env = CarRacing()
env.reset()

# Load the latest model if it exists, or start a new one
model_path = f"{models_dir}/260000.zip"  # Update this with the path to your last saved model
if os.path.exists(model_path):
    print("Loading the saved model...")
    model = PPO.load(model_path, env=env)  # Make sure to pass the environment when loading
else:
    print("No saved model found, starting a new model...")
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=logdir)
TIMESTEPS = 100000
for i in range (1,11):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{0 + i * TIMESTEPS}")

'''
env = CarRacing(render_mode="human")
env.reset()
model_path = f"{models_dir}/700000.zip"
model = PPO.load(model_path, env=env)

episodes = 10
for ep in range(episodes):
    obs, info = env.reset()
    done=False
    while not done:
        env.render()
        action,_ = model.predict(obs)
        obs, reward, done,_, info = env.step(action)

env.close()

'''