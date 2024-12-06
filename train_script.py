import os
import gymnasium as gym
from stable_baselines3 import PPO
from car_racing_modified import CarRacing
#from car_racing_original import CarRacing

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

'''
# Create the environment
env = CarRacing()
env.reset()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
TIMESTEPS = 1000
for i in range (1,31):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{i * TIMESTEPS}")

'''

env = CarRacing(render_mode="human")
env.reset()
model_path = f"{models_dir}/30000.zip"
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