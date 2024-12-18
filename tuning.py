import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from car_racing_original import CarRacing
import optuna


# Function to create the environment
def create_env():
    return CarRacing(continuous=False)


# Objective function for Optuna
def objective(trial):
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-4, 1e-2)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

    # Create the environment
    env = create_env()

    # Initialize the model with trial parameters
    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        tensorboard_log=logdir,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        gamma=gamma,
        n_steps=n_steps,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
    )

    # Train the model for a fixed number of timesteps
    model.learn(total_timesteps=100000)

    # Evaluate the model's performance
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    # Close the environment
    env.close()

    return mean_reward


# Set directories for models and logs
models_dir = "models/PPO_Optimized"
logdir = "logs_PPO_Optimized"

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Print the best hyperparameters
print("Best hyperparameters:", study.best_params)

# Train a final model with the best hyperparameters
best_params = study.best_params
env = create_env()

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=logdir,
    **best_params,
)

# Train the final model
model.learn(total_timesteps=1000000, tb_log_name="PPO_Optimized")

# Save the final model
model.save(f"{models_dir}/final_model")

env.close()
