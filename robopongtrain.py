import gym
from gym import spaces
import numpy as np
from typing import Optional
from gym.envs.registration import register
from gym.envs.classic_control import robopongenv
import pygame
from sys import exit
from stable_baselines3 import DQN
import os
from stable_baselines3 import A2C

models_dir = "models/DQN-2001"
logdir = "logs-2001"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

TIMESTEPS = 12000000

env = gym.make("robopongenv-v0")
env.reset(return_info=False)
# exploration_final_eps=(1/TIMESTEPS)

model = DQN("MultiInputPolicy", env, verbose=1, exploration_initial_eps=0.9999999999999999999999,
            exploration_final_eps=0.000000000001, target_update_interval=4,
            tensorboard_log=logdir, learning_rate=0.20, )

"""
#model = DQN("MultiInputPolicy", env, verbose=1)

#model_path = f"{models_dir}/210000"

#model = DQN.load(model_path, env=env)
"""
# episodes = 100000
# model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True, tb_log_name="DQN-static-start-2000logs ")
# model.save(f"{models_dir}/{TIMESTEPS * 1}")


for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True, tb_log_name="DQN-static-start")
    model.save(f"{models_dir}/{TIMESTEPS * i}")
    # env.render()

"""
#for episode in range(episodes):
#    obs = env.reset()
#    done = False
#    while not done:
#        env.render()
#        obs, reward, done, info = env.step(env.action_space.sample())
"""
