import os
from datetime import datetime
import matplotlib.pyplot as plt
import gymnasium
import keras.layers
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.policy import LinearAnnealedPolicy
from keras import regularizers
from keras.optimizers import Adam


# keep in mind that in order for this to work on your local machine you will have to register the environment
# in the gymnasium init file

class LoadTrainedWeights(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        weight_path, dir_list = get_weights_for_model()
        counter = 0
        for lay in model.get_weights():
            counter += 1
            weight_ = np.loadtxt(os.path.join(weight_path, f"layer-{counter}.csv"), dtype=np.float32, delimiter=",")
            lay = weight_


class GetWeightsWhileTraining(keras.callbacks.Callback):

    def __init__(self, every=5):
        # save weights every (every) episodes
        assert every > 0
        super().__init__()
        self.every = every

    def on_episode_begin(self, episode, logs):
        if episode % self.every == 0:
            weights = model.get_weights()
            model.save_weights(save_model_path + f"episode {episode}")
            episode_dir_path = os.path.join(save_model_path, f"episode-{episode}")
            os.makedirs(episode_dir_path)
            counter = 0
            for w in weights:
                counter += 1
                x = np.array(w)
                np.savetxt(fname=episode_dir_path + f"\\layer-{counter}.csv", X=x, delimiter=",", fmt="%1.32f")


def get_weights_for_model():
    path_name = os.curdir
    while 1:
        dir_file_names = []
        try:
            dir_file_names = os.listdir(path_name)
        except WindowsError:
            print("Chosen option was not a folder")
            choice = input("Try again: ")
        for i, file in enumerate(dir_file_names):
            print(f"{i} - {file}")
        choice = input("Choose the folder to load, input a negative integer to exit: ")
        try:
            int(choice)
        except ValueError:
            print("Chosen option is not a valid input")
            choice = input("Try again: ")
        if int(choice) <= -1 or int(choice) >= len(dir_file_names):
            break
        else:
            choice = int(choice)
        path_name = os.path.join(path_name, dir_file_names[choice])
    return path_name, dir_file_names


def build_model(n_actions):
    model_built = Sequential()
    lr1 = .01
    lr2 = .4

    model_built.loss = keras.losses.sparse_categorical_crossentropy
    model_built.add(keras.layers.InputLayer(input_shape=np.array([2, 5], dtype=int)))
    model_built.add(Flatten())
    activation = keras.layers.activation.ELU()

    for _ in range(10):
        model_built.add(keras.layers.BatchNormalization())
        model_built.add(Dense(500, activation=activation, kernel_initializer="he_normal",
                              kernel_regularizer=regularizers.L1L2(lr1, lr2)))

    model_built.add(Dense(n_actions, activation="softmax"))  # equivalent to sigmoid activation since n=2

    return model_built


def build_agent(model_com, actions_n, warmup=True, max_eps=1, min_eps=0.01):
    # policy = MaxBoltzmannQPolicy(eps=0.01)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps", value_max=0.9999, value_min=0.01, value_test=0.5,
                                  nb_steps=STEPS_NUMBER)
    memory = SequentialMemory(limit=1_000_000, window_length=2)
    dqn_com = DQNAgent(model=model_com, memory=memory, policy=policy, nb_actions=actions_n,
                       nb_steps_warmup=0.000001 * STEPS_NUMBER * warmup, target_model_update=1e-5,
                       batch_size=16, gamma=.99, enable_double_dqn=True)
    return dqn_com


def create_filename():
    current_time = datetime.now()
    time_format = current_time.strftime("%H-%M")
    model_name = f"pong_run_{time_format}"
    return os.path.join(os.curdir, f"logs/{model_name}")


def visualize_weights():
    graph_path, dir_list = get_weights_for_model()
    n_rows = 6
    n_cols = 11
    print("n_rows: ", n_rows)
    print("n_rows: ", n_cols)
    all_subplots = []
    for ax in range(n_rows * n_cols):
        all_subplots.append(plt.subplot(n_rows, n_cols, ax + 1))
    for m in range(n_rows * n_cols):
        if os.path.exists(os.path.join(graph_path, f"layer-{m + 1}.csv")):
            weight_ = np.loadtxt(os.path.join(graph_path, f"layer-{m + 1}.csv"), dtype=np.float32, delimiter=",")
            x = np.arange(0, weight_.shape[0], 1)
            if weight_.size > x.size:
                line = all_subplots[m].plot(weight_[0], weight_[1], "bo", alpha=0.01)[0]
            else:
                line = all_subplots[m].plot(weight_, "bo", alpha=.1)[0]
        else:
            print(os.path.join(graph_path, f"layer-{m + 1}.csv"), "doesn't exist")
    plt.show()


def get_x_y_coordinates(x, y, w, z, x_array, y_array):
    if 0 < y < 400:
        y += z
    else:
        x_array.append(x)
        y_array.append(y)
        z = -z
        y += z

    if 0 < x:
        x += w
    else:
        x_array.append(x)
        y_array.append(y)
        w = -w
        x += w
    while x < 800:
        return get_x_y_coordinates(x, y, w, z, x_array, y_array)
    x_array.append(x)
    y_array.append(y)


x_start = 130
y_start = 100
direction_x = -2
direction_y = -2
x_points = [x_start]
y_points = [y_start]

path_for_episode_observations = get_weights_for_model()[0]
observation_layer = os.path.join(path_for_episode_observations, "layer-3.csv")
output_layer = os.path.join(path_for_episode_observations, "layer-66.csv")
pd_observation = pd.read_csv(observation_layer, header=None)
pd_output = pd.read_csv(output_layer, header=None)
print(pd_observation.iloc[:])
print(np.argmax(pd_output.iloc[:]))

x_start_ep = pd_observation.iloc[1].iloc[0]
y_start_ep = pd_observation.iloc[2].iloc[0]
direction_x_ep = ((x_start_ep - pd_observation.iloc[6].iloc[0]) < 0) * -2 + ((x_start_ep - pd_observation.iloc[6].iloc[0]) > 0) * 2
direction_y_ep = ((y_start_ep - pd_observation.iloc[7].iloc[0]) < 0) * -2 + ((x_start_ep - pd_observation.iloc[7].iloc[0]) > 0) * 2
x_points_ep3 = [x_start_ep]
y_points_ep3 = [y_start_ep]

print("x_direction: ", direction_x_ep, " y_direction:", direction_y_ep)

get_x_y_coordinates(x_start_ep, y_start_ep, direction_x_ep, direction_y_ep, x_points_ep3, y_points_ep3)

for i, j in zip(x_points_ep3, y_points_ep3):
    print("x-coordinate: ", i, " y-coordinate: ", j)


save_model_path = create_filename()

env = gymnasium.make("RoboPongEnv-v0")
STEPS_NUMBER = 10_000_000

actions = env.action_space.n
model = build_model(actions)
dqn = build_agent(model, actions)

dqn.compile(Adam(learning_rate=0.00025, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])
history = dqn.fit(env, nb_steps=STEPS_NUMBER, visualize=True, verbose=2, callbacks=[GetWeightsWhileTraining()])

while 1:
    env.reset()
    # dummy moves for prediction
    step_run = [[[env.step(0)[0], env.step(1)[0]]]]
    for i in range(10000):
        actions = model.predict(step_run)
        step1 = env.step(int(actions[0, 0]))
        step2 = env.step(int(actions[0, 1]))
        env.render()
        # if one of the steps returns true for terminated, then reset
        if step1[2] or step2[2]:
            env.reset()
        step_run = [[[step1[0], step2[0]]]]
    break
