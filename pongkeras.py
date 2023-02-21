import gym
from gym import spaces
import numpy as np
from typing import Optional
from gym.envs.registration import register
import pygame
from sys import exit
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.policy import MaxBoltzmannQPolicy
from rl.policy import LinearAnnealedPolicy
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.callbacks import TensorBoard
import time
import os
from datetime import datetime
from tensorflow.keras.losses import BinaryCrossentropy
from keras.callbacks import ModelCheckpoint


def build_model(states, actions):
    model = Sequential([tensorflow.keras.layers.InputLayer(input_shape=np.array([16, 5]))])
    model.loss = BinaryCrossentropy(from_logits=True)
    #model.add(Dense(800, activation='relu'))
    model.add(Dense(1600, activation='softmax'))
    #model.add(Dense(200, activation='tanh'))
    model.add(Flatten())
    model.add(Dense(actions, activation='softmax'))
    return model


STEPS_NUMBER = 1_000_000


def build_agent(model, actions):
    #policy = MaxBoltzmannQPolicy(eps=0.01)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps", value_max=0.1, value_min=0.1, value_test=0.5,
                                  nb_steps=500_000)
    memory = SequentialMemory(limit=500000, window_length=16)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions,
                   nb_steps_warmup=7000, target_model_update=1e-3, batch_size=16, gamma=.01)
    return dqn


currtime = datetime.now()
timeform = currtime.strftime("%H-%M")
MODEL_NAME = f"pong_random_start_big_step_{timeform}"
FULL_NAME = f"logs/{MODEL_NAME}"
if not os.path.exists(FULL_NAME):
    os.makedirs(FULL_NAME)

tensorboard = TensorBoard(log_dir=f'logs/{MODEL_NAME}')

env = gym.make("robopongenv-v0")

states = env.observation_space.shape
print(states)
actions = env.action_space.n
print(env.observation_space)
model = build_model(states, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=0.00025), metrics=['accuracy'])
filepath = "pong.weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# callbacks=[tensorboard]
print(dqn.recent_action)
dqn.fit(env, nb_steps=STEPS_NUMBER, visualize=True, verbose=1, callbacks=[tensorboard])
