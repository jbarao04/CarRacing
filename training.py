import os
from sys import argv
from stable_baselines3 import PPO, SAC #type: ignore
from stable_baselines3.common.env_util import make_vec_env #type: ignore
from stable_baselines3.common.vec_env import SubprocVecEnv #type: ignore
from wrapper import CustomRewardWrapper #type: ignore

env_id = "CarRacing-v3"

TIMESTEPS = 10**6
models_dir = "models"
logdir = "logs"


def latest_model(algorithm):
    models = [int(model.split(".")[0]) for model in os.listdir(f"{models_dir}/{algorithm}")]
    models.sort()
    return f"{models_dir}/{algorithm}/{models[-1]}.zip"


def train_model(algo, algo_name, policy, n_envs=os.cpu_count()):
    env = make_vec_env(env_id, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs=dict(hardcore=True), wrapper_class=CustomRewardWrapper)

    if os.path.exists(f"{models_dir}/{algo_name}"):
        if os.listdir(f"{models_dir}/{algo_name}"):

            model_path = latest_model(algo_name)
            model = algo.load(model_path, env=env)
            iters = int(int(model_path.split("/")[2].split(".")[0]) / TIMESTEPS)
        else:

            # policy_kwargs=dict(net_arch=[{"conv1d": [32, 3, 1]}, {"fc": [256]}])  CNN nn architecture

            model = algo(policy, env, verbose=1,
                         tensorboard_log=logdir)  # (hyperparameter), learning_rate=0.0001, (Neural Network Architecture change) (Neural Network Architecture change) policy_kwargs=dict(net_arch=[256,(...n_layers...), 256]))
            iters = 0
    else:
        os.makedirs(f"{models_dir}/{algo_name}")
        model = algo(policy, env, verbose=1,
                     tensorboard_log=logdir)  # (hyperparameter), learning_rate=0.0001, (Neural Network Architecture change) policy_kwargs=dict(net_arch=[256,(...n_layers...), 256])
        iters = 0

    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, progress_bar=True, reset_num_timesteps=False,
                    tb_log_name=algo_name)  # (hyperparameters) , batch_size=256, ent_coef=0.01, vf_coef=0.5, gae_lambda=0.95))
        model.save(f"{models_dir}/{algo_name}/{TIMESTEPS * iters}")
        if iters*TIMESTEPS >= 60*10**6:
            break


def main():
    try:
        if len(argv) != 2:
            raise ValueError("No arguments given. Please specify which model to train.")

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        model_type = argv[1]

        if model_type == "PPO":
            train_model(PPO, model_type, "MlpPolicy", n_envs=100)
        elif model_type == "SAC":
            train_model(SAC, model_type, "MlpPolicy", n_envs=50)

        else:
            raise ValueError("Invalid argument. Please specify a valid algorithm.")

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()