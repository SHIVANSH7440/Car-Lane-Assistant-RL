
import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.95, epsilon=1.0):
        self.q_table = {}
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size

    def get_state_key(self, state):
        return tuple(np.round(state, 2))

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if np.random.rand() < self.epsilon or state_key not in self.q_table:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)

        q_predict = self.q_table[state_key][action]
        q_target = reward + self.gamma * np.max(self.q_table[next_key])

        self.q_table[state_key][action] += self.lr * (q_target - q_predict)
