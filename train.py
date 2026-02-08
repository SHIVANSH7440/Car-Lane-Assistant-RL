
import gym
from envs.lane_env import LaneKeepingEnv
from agent.q_learning import QLearningAgent

env = LaneKeepingEnv()
agent = QLearningAgent(state_size=3, action_size=3)

episodes = 500

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward

        if done:
            break

    agent.epsilon = max(0.01, agent.epsilon * 0.995)
    print(f"Episode {ep+1}: Reward = {total_reward:.2f}")
