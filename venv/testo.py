import random
import gym
import numpy as np
import keras
import matplotlib.pylab as plt
from collections import deque, UserDict
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam

EPISODES = 1000

TRAINING = True
PLAYING = False

DQNScoreList = []
LSTMScoreList = []
DQNTotalScore = 0
LSTMTotalScore = 0
DQNTrainingScore = 0
LSTMTrainingScore = 0
DQNTrainingList = []
LSTMTrainingList = []




class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, prevObs = None, prevAction = None):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        states = [prevObs, state]
        act_values = self.model.predict(states)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        prev_state = null
        prev_targ = null
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            if prev_targ != null:
                states = [prev_state, state]
                targets = [prev_targ, target_f]
                self.model.fit(states, targets, epochs=1, verbose=0)
            prev_state = state
            prev_targ = target_f
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class LSTMAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(LSTM(24, return_sequences=True, input_shape=(self.state_size, 1), activation='relu'))
        model.add(LSTM(24, return_sequences=True, input_shape=(self.state_size, 1), activation='relu'))
        model.add(LSTM(24))  # return a single vector of dimension 32
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, prevObs = None, prevAction = None):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.expand_dims(state, axis=2)
        states = [prevObs, state]
        act_values = self.model.predict(states)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        prev_state = None
        prev_targ = None
        for state, action, reward, next_state, done in minibatch:
            target = reward
            next_state = np.expand_dims(next_state, axis=2)
            state = np.expand_dims(state, axis=2)
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            if prev_targ != None:
                states = [prev_state, state]
                targets = [prev_targ, target_f]
                self.model.fit(states, targets, epochs=1, verbose=0)
            prev_state = state
            prev_targ = target_f
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    if TRAINING:
        env = gym.make('CartPole-v1')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        #print(state_size, action_size)
        agent = DQNAgent(state_size, action_size)
        agent1 = LSTMAgent(state_size, action_size)
        # agent.load("./save/cartpole-dqn.h5")
        done = False
        batch_size = 32

        # for e in range(EPISODES):
        #     state = env.reset()
        #     state = np.reshape(state, [1, state_size])
        #     for time in range(500):
        #         #env.render()
        #         action = agent.act(state)
        #         next_state, reward, done, _ = env.step(action)
        #         reward = reward if not done else -10
        #         next_state = np.reshape(next_state, [1, state_size])
        #         agent.remember(state, action, reward, next_state, done)
        #         state = next_state
        #         if done:
        #             DQNTrainingScore = time + DQNTrainingScore
        #             DQNTrainingList.append(DQNTrainingScore / (e+1))
        #             print(DQNTrainingScore / (e+1), e)
        #             print("episode: {}/{}, score: {}, e: {:.2}"
        #                   .format(e, EPISODES, time, agent.epsilon))
        #             break
        #     if len(agent.memory) > batch_size:
        #         agent.replay(batch_size)
        #     #if e % 10 == 0:
        # #     if e == 999:
        # #         agent.save("cartpole-dqn.h5")

        for e in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            previous_state = state
            action = agent1.act(state)
            previous_action = action
            action = agent1.act(state)
            for time in range(500):
                # env.render()
                action = agent1.act(state, previous_state, previous_action)
                next_state, reward, done, _ = env.step(action)
                # print(reward)
                reward = reward if not done else -10
                # print(reward)
                next_state = np.reshape(next_state, [1, state_size])
                # if time > 50:
                #    agent.remember(state, action, reward, next_state, done)
                agent1.remember(state, action, reward, next_state, done)
                previous_state = state
                previous_action = action
                state = next_state
                if done:
                    DQNTrainingScore = time + DQNTrainingScore
                    DQNTrainingList.append(DQNTrainingScore / (e + 1))
                    print(DQNTrainingScore / (e + 1), e)
                    print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, EPISODES, time, agent.epsilon))
                    break
            if len(agent1.memory) > batch_size:
                agent1.replay(batch_size)
            # if e % 10 == 0:
            if e == 999:
                agent1.save("cartpole-lstm.h5")


    if PLAYING:
        env = gym.make('CartPole-v1')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        # print(state_size, action_size)
        agent = DQNAgent(state_size, action_size)
        agent1 = LSTMAgent(state_size, action_size)
        agent.load("cartpole-dqn.h5")
        agent1.load("cartpole-lstm.h5")
        for e in range(10000):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            for time in range(1000):
                #env.render()
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                # print(reward)
                reward = reward if not done else -10
                # print(reward)
                next_state = np.reshape(next_state, [1, state_size])
                state = next_state
                if done:
                    DQNTotalScore = DQNTotalScore + time
                    DQNScoreList.append(time)
                    print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, 10000, time, agent.epsilon))
                    break

        for e in range(10000):
             state = env.reset()
             state = np.reshape(state, [1, state_size])
             for time in range(1000):
                 #env.render()
                 action = agent1.act(state)
                 next_state, reward, done, _ = env.step(action)
                 # print(reward)
                 reward = reward if not done else -10
                 # print(reward)
                 next_state = np.reshape(next_state, [1, state_size])
                 state = next_state
                 if done:
                     LSTMTotalScore = LSTMTotalScore + time
                     LSTMScoreList.append(time)
                     print("episode: {}/{}, score: {}, e: {:.2}"
                           .format(e, 10000, time, agent.epsilon))
                     break

        DQNTotalScore = DQNTotalScore / 10000
        LSTMTotalScore = LSTMTotalScore / 10000
        print(DQNTotalScore, LSTMTotalScore)

    fig = plt.figure()
    plt.plot(DQNTrainingList, color='red')
    plt.plot(LSTMTrainingList, color='blue')
    plt.savefig('Training2.png')
    plt.show()