
CAR LANE KEEPING USING REINFORCEMENT LEARNING (OPENAI GYM)

OVERVIEW
This project demonstrates how to design a custom OpenAI Gym environment and train a Reinforcement Learning agent
to solve a lane-keeping problem. A simplified car moves forward automatically, and the agent learns steering
behavior to stay within lane boundaries using Q-Learning implemented from scratch.

PROBLEM STATEMENT
Train an autonomous agent to:
- Stay close to the lane center
- Avoid leaving lane boundaries
- Maximize long-term reward through learning

KEY CONCEPTS USED
- Reinforcement Learning (RL)
- Markov Decision Process (MDP)
- Q-Learning (from scratch)
- Custom OpenAI Gym environment
- Reward engineering
- Exploration vs exploitation

PROJECT STRUCTURE
car-lane-rl/
|-- envs/
|   |-- lane_env.py        (Custom Gym environment)
|-- agent/
|   |-- q_learning.py     (Q-Learning implementation)
|-- train.py              (Training loop)
|-- evaluate.py           (Policy evaluation)
|-- requirements.txt
|-- README.txt

ENVIRONMENT DESIGN

Action Space:
0 -> Steer Left
1 -> Go Straight
2 -> Steer Right

Observation Space:
[ car_x_position, velocity, distance_from_lane_center ]

Reward Function:
+1.0  -> Car stays near lane center
-0.1  -> Small deviation
-10.0 -> Car leaves the lane (episode ends)

Episode Termination:
- Car goes outside lane boundaries
- Maximum steps reached

LEARNING ALGORITHM
- Tabular Q-Learning
- Implemented from scratch
- Epsilon-greedy exploration
- Discounted future rewards

TECH STACK
- Python 3
- OpenAI Gym
- NumPy

HOW TO RUN

1. Install dependencies:
pip install -r requirements.txt

2. Train the agent:
python train.py

3. Evaluate the trained policy:
python evaluate.py

EXPECTED OUTPUT
- Training rewards printed per episode
- Evaluation completes without errors

WHY THIS PROJECT MATTERS
- Shows custom environment design
- Demonstrates core RL understanding
- Relevant to game AI and vehicle control
- Easily extendable to DQN and Unreal Engine integration

FUTURE IMPROVEMENTS
- Deep Q-Network (DQN)
- Visualization
- Unreal Engine steering integration

AUTHOR
Shivansh Kumar Singh
