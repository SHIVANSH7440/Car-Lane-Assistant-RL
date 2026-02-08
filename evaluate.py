
from envs.lane_env import LaneKeepingEnv
from agent.q_learning import QLearningAgent
import numpy as np

env = LaneKeepingEnv()
agent = QLearningAgent(state_size=3, action_size=3)
agent.epsilon = 0.0

state = env.reset()
total_reward = 0

for _ in range(200):
    action = agent.choose_action(state)
    state, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        break

print("Evaluation Reward:", total_reward)
