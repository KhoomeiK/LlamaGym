import random
import gymnasium as gym
from gymnasium.spaces import Discrete, Dict, Box, Text
from llamagym import evaluate_state
from datasets import load_dataset



class QAEnv(gym.Env):
    def __init__(self):
        super(QAEnv, self).__init__()
        self.observation_space = Dict({
            'question': Text(max_length=500),
            'context': Text(max_length=500)
        })
        self.dataset = load_dataset('microsoft/orca-math-word-problems-200k')
        self.current_episode_rewards = []
        self.current_episode_messages = []

    def reset(self):
        choice = random.randint(0, len(self.dataset['train']) - 1)
        question, answer = self.dataset['train'][choice]['question'],  self.dataset['train'][choice]['answer']
        self.goal_state = answer
        return question, {}

    def step(self, action):
        agent_state = action[0]
        task = self.goal_state

        data = evaluate_state(task, agent_state, self.goal_state)
        reward = data['reward']
        info = data['feedback']
        
        print(f"Reward: {reward}")
        print(info, "green")

        done = data.get('reached_goal',False)

        return self.observation_space.sample(), reward, done,"", info

