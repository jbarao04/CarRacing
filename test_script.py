import sys
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
#from car_racing_modified import CarRacing
from car_racing_original import CarRacing
from stable_baselines3 import DDPG, PPO, DQN, A2C
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation

env = CarRacing(render_mode="human")
env = ResizeObservation(env, shape=(64,64))
env= GrayscaleObservation(env,keep_dim=False) 
env = FrameStackObservation(env, stack_size=2)

def test_model(algorithm, path):
    model = algorithm.load(path)

    mean_rew, std_rew = evaluate_policy(model, env, n_eval_episodes=50, warn=False)
    print(f"mean_reward: {mean_rew:.2f} +/- {std_rew:.2f}")

def model_testing(path, algorithm):

    if algorithm == "DQN":
        test_model(DQN, path)
    if algorithm == "A2C":
        test_model(A2C, path)
    elif algorithm == "PPO":
        test_model(PPO, path)
    elif algorithm == "DDPG":
        test_model(DDPG, path)

file_path="models/PPO/1000000.zip"
model_testing(file_path, "PPO")



