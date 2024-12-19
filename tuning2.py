import os
import gym
from stable_baselines3 import PPO
import optuna
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from car_racing_original import CarRacing

# Caminhos e configurações
models_dir = "models/PPO_Original"
model_path = os.path.join(models_dir, "500000.zip")

# Carregar o ambiente com DummyVecEnv
env = DummyVecEnv([lambda: CarRacing(continuous=True)])

# Função de avaliação do modelo
def evaluate_model(model, env, n_episodes=3):  # Reduzido para 3 episódios
    total_rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = [False]
        episode_reward = 0
        while not done[0]:  # Adaptado para DummyVecEnv
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]  # Extrair a recompensa do primeiro ambiente
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

# Objetivo para Optuna
def objective(trial):
    # Espaço de busca para hiperparâmetros principais
    learning_rate = trial.suggest_loguniform("learning_rate", 3e-4, 8e-4)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.99)

    # Carregar o modelo pré-treinado
    model = PPO.load(model_path, env=env)
    model.learning_rate = learning_rate
    model.gamma = gamma

    # Treinamento com menos timesteps
    model.learn(total_timesteps=5000)  # Reduzido ainda mais para 5,000 timesteps

    # Avaliação do modelo
    mean_reward = evaluate_model(model, env)
    return -mean_reward  # Negativo porque Optuna minimiza o valor

# Configuração e execução do estudo Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5, n_jobs=1)  # Apenas 5 trials para acelerar

# Melhor conjunto de parâmetros encontrados
best_params = study.best_params
print("Melhores Hiperparâmetros:", best_params)

# Treinar o modelo final com os melhores parâmetros
final_model = PPO(
    "CnnPolicy",
    env,
    learning_rate=best_params["learning_rate"],
    gamma=best_params["gamma"],
    verbose=1,
)
final_model.learn(total_timesteps=100000)  # Treinamento reduzido para validação rápida
final_model.save(os.path.join(models_dir, "tuned_model_fast"))

print("Modelo otimizado salvo em:", os.path.join(models_dir, "tuned_model_fast"))
